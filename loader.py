import os
import pandas as pd
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from typing import Tuple, List
from config import DEALERS_CSV_PATH, DOC_PATH, CHUNK_OVERLAP, CHUNK_SIZE, EMPLOYEE_CSV_PATH

def load_dealer_csv(path: str = DEALERS_CSV_PATH) -> Tuple[pd.DataFrame, List[Document]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dealer CSV not found at: {path}")

    df = pd.read_csv(path, dtype=str).fillna("")

    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*variants):
        for v in variants:
            key = v.lower()
            if key in cols_lower:
                return cols_lower[key]
        return None

    mappings = {
        'DEALER_ID': pick('DEALER_ID', 'dealer_id', 'id'),
        'Dealer_Name': pick('DEALER_NAME', 'dealer_name', 'name'),
        'Dealer_Brand': pick('DEALER_BRANCH', 'dealer_branch', 'branch'),
        'Dealer_Address': pick('DEALER_ADDRESS', 'dealer_address', 'address'),
        'Dealer_Pincode': pick('Pincode', 'PINCODE', 'pincode', 'pin'),
        'Dealer_City': pick('City', 'city', 'town'),
        'Dealer_State': pick('State', 'state'),
        'Dealer_Category_Level_3': pick('Dealer_Category_Level_3', 'category'),
        'FOS_Name': pick('FOS_NAME', 'fos_name'),
        'FOS_Code': pick('FOS_CODE', 'fos_code'),
        # 'Dealer_Status': pick('STATUS', 'status'),
    }

    # Rename found columns to canonical names
    rename_map = {v: k for k, v in mappings.items() if v is not None}
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure canonical columns exist
    for c in ['DEALER_ID','Dealer_Name','Dealer_Brand','Dealer_Address','Dealer_Pincode','Dealer_City','Dealer_State','Dealer_Category_Level_3','FOS_Name','FOS_Code']:
        if c not in df.columns:
            df[c] = ""

    # Strip whitespace and normalize Pincode to string (no decimals)
    df['Dealer_Pincode'] = df['Dealer_Pincode'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.replace('nan', '', regex=False).fillna("")
    df['Dealer_Category_Level_3'] = df['Dealer_Category_Level_3'].astype(str).str.strip().fillna("")
    df['Dealer_Name'] = df['Dealer_Name'].astype(str).str.strip().fillna("")
    df['Dealer_ID'] = df['DEALER_ID'].astype(str).str.strip().fillna("")
    df['Dealer_Brand'] = df['Dealer_Brand'].astype(str).str.strip().fillna("")
    df['Dealer_City'] = df['Dealer_City'].astype(str).str.strip().fillna("")
    df['Dealer_State'] = df['Dealer_State'].astype(str).str.strip().fillna("")
    df['FOS_Name'] = df['FOS_Name'].astype(str).str.strip().fillna("")
    df['FOS_Code'] = df['FOS_Code'].astype(str).str.strip().fillna("")
    # df['Dealer_Status'] = df['Dealer_Status'].astype(str).str.strip().fillna("")

    # Create a SEARCH_TEXT column for embeddings/semantic search
    # Keep order: name, category, city, branch, pincode
    def build_search_text(row):
        parts = []
        if row['Dealer_Name']:
            parts.append(row['Dealer_Name'])
        if row['Dealer_Category_Level_3']:
            parts.append(row['Dealer_Category_Level_3'])
        if row['Dealer_City']:
            parts.append(row['Dealer_City'])
        if row['Dealer_Brand']:
            parts.append(row['Dealer_Brand'])
        if row['Dealer_Pincode']:
            parts.append(row['Dealer_Pincode'])
        if row['Dealer_Address']:
            parts.append(row['Dealer_Address'])
        return " | ".join(parts)

    df['SEARCH_TEXT'] = df.apply(build_search_text, axis=1)

    # Build langchain Documents list
    docs = []
    for _, row in df.iterrows():
        content = row['SEARCH_TEXT'] or row['Dealer_Name'] or f"Dealer ID {row['Dealer_ID']}"
        metadata = {
            "pincode": row.get('Dealer_Pincode', ''),
            "category": row.get('Dealer_Category_Level_3', ''),
            "dealer_name": row.get('Dealer_Name', ''),
            "dealer_id": row.get('DEALER_ID', ''),
            "city": row.get('Dealer_City', ''),
            "state": row.get('Dealer_State', ''),
            "branch": row.get('Dealer_Brand', ''),
            "fos_name": row.get('FOS_Name', ''),
            "fos_code": row.get('FOS_Code', ''),
            # "status": row.get('Dealer_Status', '')
        }
        docs.append(Document(page_content=content, metadata=metadata))

    print(f"[loader] Loaded {len(df)} rows from {path}")
    return df, docs

def csv_to_jsonl(csv_path: str, jsonl_path: str):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[loader] Converted {csv_path} -> {jsonl_path}")

def doc_load(file_path = DOC_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: {file_path}")
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    print(f"[loader] Loaded {len(docs)} documents from {file_path}")
    return docs

def load_employee_csv(path:str = EMPLOYEE_CSV_PATH)->Tuple[pd.DataFrame, List[Document]]:
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        docs = []
        for _,i in df.iterrows():
            mob = i.get('Mob',"")
            name = i.get('Employee_Name',"")
            card = i.get('Card_Available',"")
            limit = i.get('EMI_Card_Limit',"")
            balance = i.get('Available_Balance',"")
            eligible = i.get('remi_eligible',"")
            content = f"Employee Name: {name} | EMI Card Limit: {limit} | Available Balance: {balance} | Card Available: {card} | Mobile: {mob} | REMI Eligible: {eligible}"
            docs.append(Document(page_content=content, metadata={"employee_name": name, "mobile": mob}))
        print(f"[loader] Loaded {len(df)} rows from {path}")
        return df, docs
    except Exception as e:
        print(f"[loader] Error loading employee CSV: {e}")
        return pd.DataFrame(), []

def split_docs(docs):
    split = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = split.split_documents(docs)
    print("split docs, num chunks:", len(split_docs))
    return split_docs

if __name__ == "__main__":
    # path = DEALERS_CSV_PATH if 'DEALERS_CSV_PATH' in globals() else "dealer.csv"
    # try:
    #     df, docs = load_dealer_csv(path)
    #     print(df.head(3).to_dict(orient='records'))
    #     print("Sample doc:", docs[0].page_content if docs else "no docs")
    # except Exception as e:
    #     print("Error loading dealers:", e)
    csv_to_jsonl("employee.csv", "employee.jsonl")