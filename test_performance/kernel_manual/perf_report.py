#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能报告自动化脚本
=================================

功能概述:
1. 编译: 根据 kernel_specs 中的源文件列表自动生成可执行文件 (支持环境变量 CXX/CXXFLAGS)
2. 执行: 按给定 shape(M,N,K) 调用对应的内核驱动程序 (例如 q4k_q8_main_driver.cpp), 捕获 stdout
3. 解析: 从 stdout 中提取 cycles, l1d_access, l1d_miss, miss_rate, FMA 等指标
4. 导出: 将性能数据写入 CSV (默认 results/perf_data.csv)
5. 绘图: 生成不同量化方法(quant_method) 在多个输入 shape 下的效率图 (utilization 与 每FMA的cycle)
6. 扩展: 通过在 kernel_specs 中新增条目即可加入新的 kernel; 通过 --shapes-override JSON 可一次性覆盖 shape 列表

使用示例:
	python perf_report.py --kernels all --rebuild --output results
	python perf_report.py --kernels q4k_q8 --output results --csv perf_custom.csv
	python perf_report.py --plot-only --csv perf_data.csv
	python perf_report.py --kernels all --shapes-override '{"q4k_q8": [{"M":12,"N":128,"K":1536}]}'

注意:
 - 期望内核的可执行程序通过命令行参数: <M> <N> <K>
 - 量化方法 quant_method 自动从 name 中正则提取 (例如 "q4k_q8" -> q4k & q8), 可在 spec 中覆写
 - 如果运行环境不支持 perf 事件 (读取失败) 将跳过对应指标并标记 NA
 - 如果希望使用 OpenMP, 设置环境变量 OPENMP=1 (会尝试添加 -fopenmp)
 - 若交叉编译, 设置 CXX=riscv64-unknown-linux-gnu-g++ 等

CSV 字段:
	timestamp,kernel,quant_method,M,N,K,cycles,l1d_access,l1d_miss,miss_rate,n_fma,theoretical_cycle,actual_cycle,utilization,binary

生成的图:
	utilization_vs_shape.png  : 不同 shape 下的 utilization (条形图)
	cycles_per_fma_vs_shape.png: 每 FMA 所需 cycles (越低越好)

后续可扩展点:
	- 添加多线程并行测试 (当前串行)
	- 添加更多 cache / TLB perf counters
	- 支持 JSON 输出
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

try:
	import pandas as pd  # type: ignore
	import matplotlib.pyplot as plt  # type: ignore
except ImportError:
	pd = None
	plt = None


@dataclass
class KernelSpec:
	name: str
	sources: List[str]
	binary: str
	shapes: List[Dict[str, int]]
	quant_method: str | None = None  # 可手动覆盖
	cxxflags: str | None = None      # 每个 kernel 特定编译 flags

	def resolved_quant(self) -> str:
		if self.quant_method:
			return self.quant_method
		# 简单正则: 捕获 q\d+[a-z]? 片段
		parts = re.findall(r"q\d+[a-z]?", self.name)
		return "+".join(parts) if parts else self.name

# 原始 (K, N)  权重矩阵 shape 组合
k_n_pairs = [
    (2048, 64),
    # (2048, 2048),
    # (2048, 8192),
    # (8192, 2048),
    # (3072, 128),
    # (3072, 3072),
    # (3072, 8192),
    # (8192, 3072),
	# qwen2.5-0.5b 869 不是 256 倍数 
    # (896, 64),
    # (896, 896),
    # (896, 4864),
    # (4864, 896),
    # (768, 64),
    # (768, 768),
    # (768, 4864),
    # (4864, 768),
    # (1536, 128),
    # (1536, 1536),
    # (1536, 8960),
    # (8960, 1536),
    # (2048, 128),
    # (2048, 11008),
    # (11008, 2048),
    # (6144, 128),
    # (2048, 6144),
	# gemma-3-1b-it
    # (1152, 256),
    # (1024, 1152),
    # (1152, 6912),
    # (6912, 1152),

    # (1024, 256),
    # (1024, 1024),
    # (1024, 6912),
    # (6912, 1024),
]

# M 的取值集合
# m_values = [480, 192, 48]
m_values = [192]

# 若要用于 perf_report.py (脚本里使用键名 M/N/K)
shapes_for_perf_report = []
for K, N in k_n_pairs:
    for M in m_values:
        shapes_for_perf_report.append({"M": M, "N": N, "K": K})


# 在此添加需要测试的 kernel 规格
kernel_specs: List[KernelSpec] = [
	KernelSpec(
		name="q4k_q8k_kernel",
		sources=["q4k_q8k_gemm_main_driver.cpp", "q4k_q8k_gemm_kernel.cpp"],
		binary="q4k_q8k",
		shapes=shapes_for_perf_report,
	),
	# 可继续添加其它 kernel
	KernelSpec(
		name="q40_q80_kernel",
		sources=["q40_q80_gemm_main_driver.cpp", "q40_q80_gemm_kernel.cpp"],
		binary="q40_q80",
		shapes=shapes_for_perf_report,
	),
	KernelSpec(
		name="iq4k_q8k_kernel",
		sources=["iq4k_q8k_gemm_main_driver.cpp", "iq4k_q8k_gemm_kernel.cpp"],
		binary="iq4k_q8k",
		shapes=shapes_for_perf_report,
	),
	KernelSpec(
		name="inner_q40_q80_kernel",
		sources=["inner_q40_q80_gemm_kernel_main_driver.cpp", "inner_q40_q80_gemm_kernel.cpp"],
		binary="inner_q40_q80",
		shapes=shapes_for_perf_report,
	),
	KernelSpec(
		name="inner_q4k_q8k_kernel",
		sources=["inner_q4k_q8k_gemm_kernel_main_driver.cpp", "inner_q4k_q8k_gemm_kernel.cpp"],
		binary="inner_q4k_q8k",
		shapes=shapes_for_perf_report,
	),
]


def find_spec(name: str) -> KernelSpec | None:
	for s in kernel_specs:
		if s.name == name:
			return s
	return None


def compile_kernel(spec: KernelSpec, rebuild: bool, build_dir: str) -> str:
	"""编译单个 kernel, 返回二进制路径"""
	os.makedirs(build_dir, exist_ok=True)
	binary_path = os.path.join(build_dir, spec.binary)
	if os.path.exists(binary_path) and not rebuild:
		return binary_path
	cxx = os.environ.get("CXX", "clang++")
	base_flags = os.environ.get("CXXFLAGS", "-O3 -DSPACEMIT_X60 -march=rv64gcv_zvfh")
	# 可选：DEBUG=1 降优化并加符号
	if os.environ.get("DEBUG") == "1":
		base_flags = base_flags.replace("-O3", "-O0") + " -g -fno-omit-frame-pointer"
	# 可选：SANITIZE=address 启用 AddressSanitizer
	sanitize = os.environ.get("SANITIZE", "").lower()
	if "address" in sanitize:
		base_flags += " -fsanitize=address"
	if os.environ.get("OPENMP") == "1":
		base_flags += " -fopenmp"
	if spec.cxxflags:
		base_flags += f" {spec.cxxflags}"
	cmd = [cxx] + base_flags.split() + spec.sources + ["-o", binary_path]
	print(f"[BUILD] {' '.join(cmd)}")
	try:
		subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
	except subprocess.CalledProcessError as e:
		print(f"编译失败: {spec.name}: {e}")
		return ""
	return binary_path


perf_line_patterns = {
	"cycle": re.compile(r"cycle\s*=\s*(\d+)"),
	"l1d_access": re.compile(r"l1d_access\s*=\s*(\d+)"),
	"l1d_miss": re.compile(r"l1d_miss\s*=\s*(\d+)"),
	"miss_rate": re.compile(r"miss_rate\s*=\s*([0-9.]+)%"),
	"n_fma": re.compile(r"乘加数:\s*(\d+)"),
	"theoretical_cycle": re.compile(r"理论需要周期数:\s*(\d+)"),
	"actual_cycle": re.compile(r"实际执行周期数:\s*(\d+)"),
	"utilization": re.compile(r"实际-理论比值:\s*([0-9.]+)\s*\(([0-9.]+)%\)"),
}


def parse_perf_output(out: str) -> Dict[str, Any]:
	data: Dict[str, Any] = {}
	for k, pat in perf_line_patterns.items():
		m = pat.search(out)
		if m:
			if k == "utilization":
				data["utilization_ratio"] = float(m.group(1))
				# group(2) 是百分比数值（例如 75.3），以带 '%' 的字符串形式保存到 CSV 中
				pct = float(m.group(2))
				data["utilization_percent_inverse"] = f"{pct}%"
			else:
				val = m.group(1)
				data[k] = float(val) if "." in val else int(val)
		else:
			data[k] = "NA"
	# cycles 平均: actual_cycle 已经来自脚本, cycle_total/T; 这里留存全部原始
	return data


def run_kernel(binary: str, shape: Dict[str, int]) -> str:
	if not os.path.exists(binary):
		return ""
	cmd = [binary, str(shape["M"]), str(shape["N"]), str(shape["K"])]
	print(f"[RUN ] {' '.join(cmd)}")
	try:
		res = subprocess.run(cmd, cwd=os.path.dirname(binary), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
		return res.stdout
	except subprocess.CalledProcessError as e:
		print(f"运行失败: {binary}: {e}\n输出: {e.stdout}")
		return e.stdout or ""


def ensure_dependencies():
	if pd is None or plt is None:
		print("缺少 pandas 或 matplotlib. 请先安装: pip install pandas matplotlib")
		return False
	return True


def load_existing(csv_path: str) -> List[Dict[str, Any]]:
	if not os.path.exists(csv_path):
		return []
	with open(csv_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		return list(reader)


def save_csv(rows: List[Dict[str, Any]], csv_path: str):
	if not rows:
		return
	fieldnames = [
		"timestamp","kernel","quant_method","M","N","K","cycles","l1d_access","l1d_miss","miss_rate","n_fma","theoretical_cycle","actual_cycle","utilization_percent_inverse","binary"
	]
	need_header = not os.path.exists(csv_path)
	with open(csv_path, "a" if not need_header else "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if need_header:
			writer.writeheader()
		for r in rows:
			writer.writerow({k: r.get(k, "") for k in fieldnames})
	print(f"[CSV ] 写入 {len(rows)} 行 -> {csv_path}")


def make_plots(csv_path: str, output_dir: str):
	if not ensure_dependencies():
		return
	df = pd.read_csv(csv_path)
	# 形状标签
	df["shape"] = df.apply(lambda r: f"M{r['M']}-N{r['N']}-K{r['K']}", axis=1)
	# 将百分比字符串转换为数值 (去掉 '%' 符号)
	df["utilization_pct"] = df["utilization_percent_inverse"].apply(lambda x: float(str(x).rstrip('%')) if pd.notna(x) and str(x) != 'NA' else float('nan'))
	# 利用率图
	plt.figure(figsize=(10, 5))
	for qm, sub in df.groupby("quant_method"):
		plt.plot(sub["shape"], sub["utilization_pct"], marker="o", label=qm)
	plt.xticks(rotation=30, ha="right")
	plt.ylabel("utilization (%)")
	plt.title("Kernel Utilization vs Shape")
	plt.legend()
	plt.tight_layout()
	util_path = os.path.join(output_dir, "utilization_vs_shape.png")
	plt.savefig(util_path, dpi=150)
	print(f"[PLOT] 保存 {util_path}")
	plt.close()

	# 每 FMA cycles (cycle_per_fma = actual_cycle / n_fma) 越低越好
	df["cycle_per_fma"] = df.apply(lambda r: (float(r["actual_cycle"]) / float(r["n_fma"])) if str(r["actual_cycle"]).isdigit() and str(r["n_fma"]).isdigit() else float("nan"), axis=1)
	plt.figure(figsize=(10, 5))
	for qm, sub in df.groupby("quant_method"):
		plt.plot(sub["shape"], sub["cycle_per_fma"], marker="s", label=qm)
	plt.xticks(rotation=30, ha="right")
	plt.ylabel("cycles per FMA")
	plt.title("Cycles per FMA vs Shape")
	plt.legend()
	plt.tight_layout()
	cpf_path = os.path.join(output_dir, "cycles_per_fma_vs_shape.png")
	plt.savefig(cpf_path, dpi=150)
	print(f"[PLOT] 保存 {cpf_path}")
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="性能报告生成脚本")
	parser.add_argument("--kernels", default="all", help="逗号分隔的 kernel 名称或 all")
	parser.add_argument("--rebuild", action="store_true", help="强制重新编译")
	parser.add_argument("--output", default="results", help="输出目录")
	parser.add_argument("--csv", default="perf_data.csv", help="CSV 文件名")
	parser.add_argument("--plot-only", action="store_true", help="仅根据已有 CSV 生成图")
	parser.add_argument("--shapes-override", help="JSON 格式: {\"kernel\": [{M,N,K},...]} 覆盖默认 shape")
	args = parser.parse_args()

	script_dir = os.path.dirname(__file__)
	output_dir = os.path.join(script_dir, args.output)
	os.makedirs(output_dir, exist_ok=True)
	csv_path = os.path.join(output_dir, args.csv)

	if args.shapes_override:
		override = json.loads(args.shapes_override)
		for kname, shapes in override.items():
			spec = find_spec(kname)
			if spec:
				spec.shapes = shapes

	selected: List[KernelSpec]
	if args.kernels == "all":
		selected = kernel_specs
	else:
		names = [x.strip() for x in args.kernels.split(",") if x.strip()]
		selected = [s for s in kernel_specs if s.name in names]
		missing = set(names) - {s.name for s in selected}
		if missing:
			print(f"未找到 kernel: {', '.join(missing)}")

	rows: List[Dict[str, Any]] = []
	build_dir = output_dir  # 将二进制也放入 output 目录

	if not args.plot_only:
		for spec in selected:
			binary = compile_kernel(spec, args.rebuild, build_dir)
			if not binary:
				continue
			for shape in spec.shapes:
				out = run_kernel(binary, shape)
				if not out:
					continue
				perf = parse_perf_output(out)
				row: Dict[str, Any] = {
					"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
					"kernel": spec.name,
					"quant_method": spec.resolved_quant(),
					"M": shape["M"],
					"N": shape["N"],
					"K": shape["K"],
					"binary": os.path.basename(binary),
				}
				row.update(perf)
				rows.append(row)
		save_csv(rows, csv_path)

	if os.path.exists(csv_path):
		make_plots(csv_path, output_dir)
	else:
		print("没有找到 CSV，无法绘图。")


if __name__ == "__main__":
	main()

