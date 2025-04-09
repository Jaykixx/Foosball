import torch
import csv


def write_csv(filepath, tensor_dict, delimiter=','):
    headers = tensor_dict.keys()
    rows = torch.stack(list(tensor_dict.values()), dim=-1).cpu()
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row.tolist())


def read_csv(filepath, delimiter=',', device='cpu'):
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader)
        rows = []
        for row in reader:
            if row:
                rows.append(
                    torch.tensor([float(value) for value in row], device=device)
                )
    rows = torch.stack(rows, dim=-1)
    data = {}
    for idx, key in enumerate(headers):
        data[key] = rows[idx]

    return data
