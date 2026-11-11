"""Unit tests for BTS cleaning logic."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bts_clean import clean_bts_data, get_spark_session


class TestBTSCleaning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_spark_session("test_bts_cleaning")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_cleaning_rules(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bronze = tmp_path / "bronze" / "2024" / "01"
            bronze.mkdir(parents=True, exist_ok=True)
            silver = tmp_path / "silver"
            results = tmp_path / "results"

            csv_file = bronze / "sample.csv"
            # Row 1: Valid on-time
            # Row 2: Cancelled (should be excluded)
            # Row 3: Diverted (should be excluded)
            # Row 4: Missing target (should be excluded)
            # Row 5: Missing flight number (should be excluded)
            # Row 6: Valid delayed
            lines = [
                "FlightDate,Reporting_Airline,Flight_Number_Reporting_Airline,Origin,Dest,CRSDepTime,CRSArrTime,CRSElapsedTime,Distance,Cancelled,Diverted,ArrDel15\n",
                "2024-01-01, AA , 101 , JFK , LAX , 600 , 930 , 330.0 , 2475.0 , 0.0 , 0.0 , 0.0 \n",
                "2024-01-01, AA , 102 , JFK , LAX , 700 , 1030 , 330.0 , 2475.0 , 1.0 , 0.0 ,     \n",
                "2024-01-01, AA , 103 , JFK , LAX , 800 , 1130 , 330.0 , 2475.0 , 0.0 , 1.0 ,     \n",
                "2024-01-01, AA , 104 , JFK , LAX , 900 , 1230 , 330.0 , 2475.0 , 0.0 , 0.0 ,     \n",
                "2024-01-01, AA ,     , JFK , LAX , 1000 , 1330 , 330.0 , 2475.0 , 0.0 , 0.0 , 0.0 \n",
                "2024-01-01, DL , 201 , ATL , ORD , 1200 , 1315 , 135.0 , 606.0 , 0.0 , 0.0 , 1.0 \n",
            ]
            csv_file.write_text("".join(lines), encoding="utf-8")

            summary = clean_bts_data(
                spark=self.spark,
                bronze_dir=tmp_path / "bronze",
                silver_dir=silver,
                results_dir=results,
            )

            # Check summary counts
            self.assertEqual(len(summary), 1)
            row = summary.iloc[0]
            self.assertEqual(row["raw_records"], 6)
            self.assertEqual(row["excluded_cancelled"], 1)
            self.assertEqual(row["excluded_diverted"], 1)
            self.assertEqual(row["excluded_missing_flight_num"], 1)
            self.assertEqual(row["retained_records"], 2)

            # Check that folder structure matches bronze (2024/01)
            month_silver_dir = silver / "2024" / "01"
            self.assertTrue(month_silver_dir.exists())

            # Check cleaned silver dataset reading
            cleaned_df = self.spark.read.parquet(str(silver / "*/*"))
            self.assertEqual(cleaned_df.count(), 2)

            # Verify deterministic flight ID and left padding of time
            records = {r["Reporting_Airline"]: r for r in cleaned_df.collect()}
            self.assertEqual(records["AA"]["flight_id"], "2024-01-01_AA_101_JFK_LAX_0600")
            self.assertEqual(records["AA"]["CRSDepTime"], "0600")
            self.assertEqual(records["AA"]["ArrDel15"], 0)
            self.assertEqual(records["DL"]["ArrDel15"], 1)


if __name__ == "__main__":
    unittest.main()
