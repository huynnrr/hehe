"""
format_benchmark.py
--------------------
Task 2 - Part B: File Format Benchmarking & Partition Management

Muc tieu:
  1. Doc du lieu chuyen bay (Kaggle "2015 Flight Delays and Cancellations").
  2. Ghi ra 4 dinh dang: CSV, JSON, Parquet, ORC.
  3. Do va ghi lai:
       - Dung luong dia (Disk Storage Size)
       - Thoi gian ghi (Write Execution Time)
       - Thoi gian doc truy van cot (Read Query Time) cho cau lenh:
             SELECT AIRLINE, avg(ARRIVAL_DELAY) FROM ... GROUP BY AIRLINE
  4. Thi nghiem kiem soat partition:
       - .repartition(20) vs .coalesce(2) khi ghi Parquet
       - .partitionBy("YEAR", "MONTH")
  5. Xuat toan bo ket qua benchmark ra 1 file CSV (benchmark_results.csv)
     de dan vao REPORT.md.

Cach chay KHUYEN NGHI (dung spark-submit de driver-memory duoc ap dung dung cach):
    spark-submit --driver-memory 4g --master local[2] format_benchmark.py \
        --input-path ./data/flights.csv --output-path ./data/output

Test nhanh voi mau nho truoc khi chay full (kiem tra logic, KHONG dung so lieu
nay de ket luan ve hieu nang - xem muc "Debugging thuc te" trong REPORT.md):
    spark-submit --driver-memory 2g --master local[2] format_benchmark.py \
        --input-path ./data/flights.csv --output-path ./data/output --sample-fraction 0.01

Neu may RAM thap va van OOM khi chay full data, tang --write-partitions
(vd 20, 40) thay vi giam - xem giai thich chi tiet trong comment ben duoi.
"""

import argparse
import os
import shutil
import time

# ---------------------------------------------------------------------------
# QUAN TRONG: phai set truoc khi import pyspark!
# Khi chay script bang "python format_benchmark.py" (khong qua spark-submit),
# day la CACH DUY NHAT de tang driver memory, vi JVM cua Spark khoi dong
# NGAY LUC pyspark duoc import lan dau. Neu set qua
# SparkSession.builder.config("spark.driver.memory", ...) thi da qua muon
# (JVM da chay voi heap mac dinh ~1GB), day chinh la nguyen nhan gay loi
# "OutOfMemoryError: Java heap space" khi ghi CSV/shuffle du lieu lon.
#
# Neu chay qua spark-submit (khuyen nghi - xem submit_job.sh cua Task 3),
# dong nay se bi bo qua vi PYSPARK_SUBMIT_ARGS da duoc dat boi flag
# --driver-memory tren dong lenh, va do phai chi dinh o do thay vi o day.
if "PYSPARK_SUBMIT_ARGS" not in os.environ:
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--driver-memory 4g --conf spark.driver.maxResultSize=2g pyspark-shell"
    )

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_dir_size_bytes(path: str) -> int:
    """Cong don kich thuoc tat ca cac file thuc te trong 1 thu muc output cua Spark.

    Spark ghi ra nhieu file part-xxxxx.snappy.parquet / .json / .csv... ben trong
    1 thu muc, nen ta phai duyet toan bo thu muc (bao gom ca cac thu muc con khi
    dung partitionBy) roi cong tong dung luong lai.
    """
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.startswith("_") or f.endswith(".crc"):
                continue
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def human_readable(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def count_files(path: str) -> int:
    """Dem so luong file du lieu thuc te (bo qua _SUCCESS, .crc) - dung de minh
    hoa 'Small File Problem' khi so sanh repartition vs coalesce."""
    cnt = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if not f.startswith("_") and not f.endswith(".crc"):
                cnt += 1
    return cnt


def clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


class Benchmark:
    """Gom toan bo ket qua do dac lai thanh 1 bang, roi xuat ra CSV cho REPORT.md."""

    def __init__(self):
        self.rows = []  # list of dict

    def add(self, stage, name, **metrics):
        row = {"stage": stage, "name": name}
        row.update(metrics)
        self.rows.append(row)
        print(f"[BENCHMARK] {stage:<20} | {name:<20} | {metrics}")

    def to_csv(self, path):
        # Thu thap toan bo ten cot xuat hien trong cac dong (moi stage co the co
        # cot metric khac nhau: write_time vs read_time vs num_files...)
        fieldnames = ["stage", "name"]
        for row in self.rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spark File Format & Partition Benchmark")
    parser.add_argument("--input-path", required=True, help="Duong dan file flights.csv dau vao")
    parser.add_argument("--output-path", required=True, help="Thu muc goc de ghi cac output benchmark")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Ty le lay mau du lieu (0-1). Dung <1.0 de chay nhanh/nhe RAM hon khi test tren may ca nhan.",
    )
    parser.add_argument(
        "--local-cores",
        type=int,
        default=2,
        help="So core dung cho local mode. Giam so nay neu may bi OOM (it task song song hon "
        "= it buffer shuffle tranh nhau RAM hon). Mac dinh 2 (an toan cho may RAM thap).",
    )
    parser.add_argument(
        "--driver-memory",
        type=str,
        default="4g",
        help="Chi de HIEN THI goi y - khong co tac dung thuc te khi chay bang 'python script.py'. "
        "De thuc su ap dung, PHAI truyen qua dong lenh: "
        "spark-submit --driver-memory 4g format_benchmark.py ...",
    )
    parser.add_argument(
        "--write-partitions",
        type=int,
        default=8,
        help="So partition khi ghi CSV/JSON/Parquet/ORC (buoc benchmark dinh dang). "
        "QUAN TRONG: so cang NHO thi moi partition cang phai giu CANG NHIEU du lieu trong RAM "
        "khi Spark shuffle-sort truoc khi ghi -> de OOM tren may RAM thap. "
        "Neu van OOM, HAY TANG so nay len (vd 20, 40) thay vi giam.",
    )
    args = parser.parse_args()

    print(
        f"[INFO] Neu ban thay canh bao nay, PYSPARK_SUBMIT_ARGS co the KHONG duoc ap dung "
        f"tren he thong cua ban. Cach chac chan nhat de dat driver memory la chay qua:\n"
        f"    spark-submit --driver-memory {args.driver_memory} --master local[{args.local_cores}] "
        f"format_benchmark.py --input-path ... --output-path ...\n"
    )

    spark = (
        SparkSession.builder.appName("FileFormatBenchmark")
        .master(f"local[{args.local_cores}]")
        .config("spark.sql.shuffle.partitions", str(max(10, args.write_partitions)))
        .config("spark.driver.memory", args.driver_memory)  # co the khong co tac dung (xem ghi chu tren)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    bench = Benchmark()

    # -----------------------------------------------------------------
    # 1) Doc du lieu goc (CSV co header, de Spark tu infer schema)
    # -----------------------------------------------------------------
    t0 = time.time()
    df = spark.read.csv(args.input_path, header=True, inferSchema=True)
    if args.sample_fraction < 1.0:
        df = df.sample(fraction=args.sample_fraction, seed=42)
    df = df.cache()
    row_count = df.count()
    print(f"Loaded {row_count:,} rows in {time.time() - t0:.2f}s")

    # Giữ nguyên toàn bộ 31 cột của dataset gốc cho benchmark.
    # Query đọc bên dưới vẫn chỉ chọn AIRLINE và ARRIVAL_DELAY.
    df_clean = df

    formats_dir = os.path.join(args.output_path, "formats")
    clean_dir(formats_dir)

    # -----------------------------------------------------------------
    # 2) Ghi ra 4 dinh dang va do Write Time + Disk Size
    # -----------------------------------------------------------------
    format_writers = {
        "csv": lambda writer, path: writer.option("header", True).csv(path),
        "json": lambda writer, path: writer.json(path),
        "parquet": lambda writer, path: writer.parquet(path),
        "orc": lambda writer, path: writer.orc(path),
    }

    for fmt, write_fn in format_writers.items():
        out_path = os.path.join(formats_dir, fmt)
        clean_dir(out_path)

        # QUAN TRONG: dung args.write_partitions (mac dinh 8) thay vi ep cung ve 4.
        # Neu chia thanh qua it partition, moi partition phai giu qua nhieu du lieu
        # trong buffer RAM luc Spark shuffle-sort truoc khi ghi -> gay OutOfMemoryError
        # tren may cau hinh thap, du la Driver OOM hay Executor OOM (Chuong 18).
        writer = df_clean.repartition(args.write_partitions).write.mode("overwrite")

        t0 = time.time()
        write_fn(writer, out_path)
        write_time = time.time() - t0

        size_bytes = get_dir_size_bytes(out_path)
        bench.add(
            "write",
            fmt.upper(),
            write_time_sec=round(write_time, 3),
            size_bytes=size_bytes,
            size_human=human_readable(size_bytes),
        )

    # -----------------------------------------------------------------
    # 3) Read Query Time: SELECT AIRLINE, avg(ARRIVAL_DELAY) ... GROUP BY AIRLINE
    #    Doc lai tu chinh 4 dinh dang vua ghi, do thoi gian truy van cot.
    # -----------------------------------------------------------------
    readers = {
        "csv": lambda path: spark.read.csv(path, header=True, inferSchema=True),
        "json": lambda path: spark.read.json(path),
        "parquet": lambda path: spark.read.parquet(path),
        "orc": lambda path: spark.read.orc(path),
    }

    for fmt, read_fn in readers.items():
        in_path = os.path.join(formats_dir, fmt)

        t0 = time.time()
        rdf = read_fn(in_path)
        result = (
            rdf.select("AIRLINE", "ARRIVAL_DELAY")
            .groupBy("AIRLINE")
            .agg(avg("ARRIVAL_DELAY").alias("avg_arrival_delay"))
        )
        # .collect() de bat buoc Spark thuc thi thuc su (do lazy evaluation,
        # neu khong goi action thi Spark se khong chay gi ca)
        result.collect()
        read_time = time.time() - t0

        bench.add("read_query", fmt.upper(), read_time_sec=round(read_time, 3))

    # -----------------------------------------------------------------
    # 4) Partition Control Experiment
    # -----------------------------------------------------------------
    partition_dir = os.path.join(args.output_path, "partition_experiment")
    clean_dir(partition_dir)

    # 4a. repartition(20): tang so partition -> tang so file khi ghi
    repart_path = os.path.join(partition_dir, "repartition_20")
    t0 = time.time()
    df_clean.repartition(20).write.mode("overwrite").parquet(repart_path)
    t_repart = time.time() - t0
    bench.add(
        "partition_control",
        "repartition(20)",
        write_time_sec=round(t_repart, 3),
        num_files=count_files(repart_path),
        size_bytes=get_dir_size_bytes(repart_path),
        size_human=human_readable(get_dir_size_bytes(repart_path)),
    )

    # 4b. coalesce(2): giam so partition -> gop lai thanh it file hon,
    #     KHONG gay shuffle toan bo (khac voi repartition).
    coalesce_path = os.path.join(partition_dir, "coalesce_2")
    t0 = time.time()
    df_clean.coalesce(2).write.mode("overwrite").parquet(coalesce_path)
    t_coalesce = time.time() - t0
    bench.add(
        "partition_control",
        "coalesce(2)",
        write_time_sec=round(t_coalesce, 3),
        num_files=count_files(coalesce_path),
        size_bytes=get_dir_size_bytes(coalesce_path),
        size_human=human_readable(get_dir_size_bytes(coalesce_path)),
    )

    # 4c. partitionBy("YEAR", "MONTH"): ghi theo thu muc phan cap
    #     YEAR=2015/MONTH=1/, YEAR=2015/MONTH=2/, ... -> ho tro predicate
    #     pushdown / partition pruning khi query WHERE YEAR=2015 AND MONTH=1
    partitionby_path = os.path.join(partition_dir, "partition_by_year_month")
    t0 = time.time()
    df_clean.write.mode("overwrite").partitionBy("YEAR", "MONTH").parquet(partitionby_path)
    t_partitionby = time.time() - t0

    # Liet ke cac thu muc con duoc tao ra de minh hoa trong REPORT.md
    sample_subdirs = []
    for root, dirs, _files in os.walk(partitionby_path):
        for d in dirs:
            sample_subdirs.append(os.path.relpath(os.path.join(root, d), partitionby_path))
    sample_subdirs = sorted(sample_subdirs)[:5]

    bench.add(
        "partition_control",
        "partitionBy(YEAR,MONTH)",
        write_time_sec=round(t_partitionby, 3),
        num_files=count_files(partitionby_path),
        size_bytes=get_dir_size_bytes(partitionby_path),
        size_human=human_readable(get_dir_size_bytes(partitionby_path)),
        sample_subdirs=";".join(sample_subdirs),
    )

    # -----------------------------------------------------------------
    # 5) Xuat ket qua
    # -----------------------------------------------------------------
    results_csv = os.path.join(args.output_path, "benchmark_results.csv")
    bench.to_csv(results_csv)
    print(f"\nDa ghi ket qua benchmark vao: {results_csv}")

    spark.stop()


if __name__ == "__main__":
    main()