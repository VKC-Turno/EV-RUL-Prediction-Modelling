"""Smoke-test the deployed per-OEM SoH endpoints.

For each OEM endpoint that exists, send a minimal featengg-shaped record and assert the response carries a
q50 SoH forecast in [0, 100]. Fails the build (nonzero exit) if any live endpoint misbehaves — that gate
keeps a broken model from being promoted staging -> prod.
"""
import argparse
import json
import os
import sys

import boto3

OEMS = os.environ.get("OEMS", "euler mahindra bajaj piaggio montra").split()

SAMPLE = {"instances": [{"soh": 96.0, "age_months": 8, "km_month": 1200, "horizon": 6}]}


def test_endpoint(rt, name):
    resp = rt.invoke_endpoint(EndpointName=name, ContentType="application/json",
                              Body=json.dumps(SAMPLE))
    payload = json.loads(resp["Body"].read())
    q50 = payload.get("q50") or payload.get("predictions", [{}])[0].get("q50")
    assert q50 is not None, f"{name}: no q50 in response {payload}"
    assert 0.0 <= float(q50) <= 100.0, f"{name}: q50 out of range: {q50}"
    print(f"[OK] {name}: q50={q50}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default=os.environ.get("AWS_REGION"))
    p.add_argument("--stage", default="staging")
    a = p.parse_args()

    sm = boto3.client("sagemaker", region_name=a.region)
    rt = boto3.client("sagemaker-runtime", region_name=a.region)
    live = {e["EndpointName"] for e in sm.list_endpoints(MaxResults=100)["Endpoints"]}

    failures = []
    for oem in OEMS:
        name = f"soh-{oem}-{a.stage}"
        if name not in live:
            print(f"[skip] {name} not deployed")
            continue
        try:
            test_endpoint(rt, name)
        except Exception as e:            # noqa: BLE001 — collect all, fail at the end
            print(f"[FAIL] {name}: {e}")
            failures.append(name)

    if failures:
        sys.exit(f"endpoint smoke-test failed: {failures}")
    print("all live endpoints passed")


if __name__ == "__main__":
    main()
