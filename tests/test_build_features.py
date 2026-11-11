"""Unit tests for feature engineering and chronological split assignment."""

from pathlib import Path
import sys
import tempfile
import unittest

from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bts_transform import build_features, get_spark_session


class TestBuildFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_spark_session("test_build_features")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_feature_derivation_and_splitting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            silver = tmp_path / "silver"
            gold = tmp_path / "gold"
            results = tmp_path / "results"

            # Create synthetic rows across train, val, and test dates
            # 2024-01-01 is Monday (DayOfWeek=1 in BTS convention)
            # 2024-01-07 is Sunday (DayOfWeek=7 in BTS convention)
            # 2024-10-15 is in validation window
            # 2025-01-10 is in test window
            lines = [
                "flight_id,FlightDate,Reporting_Airline,Flight_Number_Reporting_Airline,Origin,Dest,CRSDepTime,CRSArrTime,CRSElapsedTime,Distance,ArrDel15,Cancelled,Diverted,source_file,Year,Month\n",
                "2024-01-01_AA_101_JFK_LAX_0600,2024-01-01,AA,101,JFK,LAX,0600,0930,330.0,2475.0,0,0,0,test.csv,2024,1\n",
                "2024-01-07_DL_202_ATL_ORD_2400,2024-01-07,DL,202,ATL,ORD,2400,0115,135.0,606.0,1,0,0,test.csv,2024,1\n",
                "2024-10-15_UA_303_SFO_DEN_1415,2024-10-15,UA,303,SFO,DEN,1415,1745,150.0,967.0,0,0,0,test.csv,2024,10\n",
                "2025-01-10_WN_404_MDW_BWI_2130,2025-01-10,WN,404,MDW,BWI,2130,0020,110.0,607.0,1,0,0,test.csv,2025,1\n",
            ]

            csv_dir = tmp_path / "csv_input"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_file = csv_dir / "sample.csv"
            csv_file.write_text("".join(lines), encoding="utf-8")

            # Write to partitioned silver directory via Spark
            raw_df = (
                self.spark.read.csv(str(csv_file), header=True)
                .withColumn("FlightDate", F.to_date("FlightDate", "yyyy-MM-dd"))
                .withColumn("CRSElapsedTime", F.col("CRSElapsedTime").cast("double"))
                .withColumn("Distance", F.col("Distance").cast("double"))
                .withColumn("ArrDel15", F.col("ArrDel15").cast("int"))
                .withColumn("Cancelled", F.col("Cancelled").cast("int"))
                .withColumn("Diverted", F.col("Diverted").cast("int"))
                .withColumn("Year", F.col("Year").cast("int"))
                .withColumn("Month", F.col("Month").cast("int"))
            )
            silver_month = silver / "2024" / "01"
            silver_month.mkdir(parents=True, exist_ok=True)
            raw_df.write.mode("overwrite").parquet(str(silver_month))

            summary = build_features(
                spark=self.spark,
                silver_dir=silver,
                gold_dir=gold,
                results_dir=results,
            )

            # Check summary dataframe
            self.assertEqual(len(summary), 3)
            self.assertListEqual(summary["split"].tolist(), ["train", "val", "test"])
            train_row = summary[summary["split"] == "train"].iloc[0]
            val_row = summary[summary["split"] == "val"].iloc[0]
            test_row = summary[summary["split"] == "test"].iloc[0]

            self.assertEqual(train_row["total_flights"], 2)
            self.assertEqual(val_row["total_flights"], 1)
            self.assertEqual(test_row["total_flights"], 1)
            self.assertEqual(train_row["delayed_flights"], 1)
            self.assertEqual(train_row["on_time_flights"], 1)
            self.assertEqual(train_row["delay_rate"], 0.5)
            self.assertEqual(val_row["delayed_flights"], 0)
            self.assertEqual(test_row["delayed_flights"], 1)

            # Verify saved gold partitions
            train_df = self.spark.read.parquet(str(gold / "train"))
            val_df = self.spark.read.parquet(str(gold / "val"))
            test_df = self.spark.read.parquet(str(gold / "test"))

            self.assertEqual(train_df.count(), 2)
            self.assertEqual(val_df.count(), 1)
            self.assertEqual(test_df.count(), 1)

            train_records = {r["flight_id"]: r for r in train_df.collect()}
            # AA flight: 2024-01-01 (Monday), dep 0600 -> hour 6, arr 0930 -> hour 9
            aa = train_records["2024-01-01_AA_101_JFK_LAX_0600"]
            self.assertEqual(aa["Month"], 1)
            self.assertEqual(aa["DayOfWeek"], 1)
            self.assertEqual(aa["DepHour"], 6)
            self.assertEqual(aa["ArrHour"], 9)
            self.assertEqual(aa["ArrDel15"], 0)

            # DL flight: 2024-01-07 (Sunday -> DayOfWeek=7), dep 2400 -> hour 0, arr 0115 -> hour 1
            dl = train_records["2024-01-07_DL_202_ATL_ORD_2400"]
            self.assertEqual(dl["DayOfWeek"], 7)
            self.assertEqual(dl["DepHour"], 0)
            self.assertEqual(dl["ArrHour"], 1)
            self.assertEqual(dl["ArrDel15"], 1)


if __name__ == "__main__":
    unittest.main()
