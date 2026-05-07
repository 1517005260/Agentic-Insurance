"""Pydantic v2 request / response shapes for the web API.

Kept thin and intent-revealing — one file per route group. ORM rows
travel through these for `/openapi.json` generation; the frontend's
``openapi-typescript`` codegen depends on the field names being stable.
"""
