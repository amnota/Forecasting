import pandas as pd
from fastapi import UploadFile
from io import BytesIO

async def read_file(file: UploadFile):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        return df, None
    except Exception as e:
        return None, f"Error reading CSV file: {e}"
