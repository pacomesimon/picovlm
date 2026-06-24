import os

def batch_iterable(iterable, batch_size):
    """Yield successive batches from iterable"""
    it = list(iterable).copy()
    for i in range(0, len(it), batch_size):
        yield it[i:i + batch_size]

def zip_folder(folder_path):
    """
    Reads all .txt files in the folder and returns them as a list of dictionaries.
    Wrapper name preserved from original code, though it returns content list.
    """
    if (len(folder_path) == 0) or (not os.path.exists(folder_path)):
        return None

    # Original code had references to creating a csv/zip but commented out.
    # We only reproduce the active logic.
    data = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".txt"):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            data.append({"filename": fname, "content": content})

    return data
