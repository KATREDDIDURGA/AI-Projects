import pandas as pd
import random
import os

# Simple transaction dataset generator

user_queries = [
    "I didn’t receive my order",
    "Order was missing",
    "Wrong product received",
    "Item is damaged",
    "Never got tracking update",
    "Need a refund urgently",
    "Delivered but empty box",
    "I want a refund for last week’s order",
    "No delivery, please refund",
    "Package lost in transit",
]

items = [
    "Wireless Earbuds",
    "Smartphone Cover",
    "Bluetooth Speaker",
    "Gaming Mouse",
    "Laptop Stand",
    "Noise Cancelling Headphones",
    "USB-C Charger",
    "Smartwatch Strap",
    "Portable SSD",
    "Webcam",
]

def generate_dataset(n_rows=600):
    data = []
    for i in range(n_rows):
        transaction_id = f"T{i+1:04d}"
        query = random.choice(user_queries)
        item = random.choice(items)
        amount = round(random.uniform(20, 300), 2)
        prior_fraud_flag = random.choice(["Yes", "No"])
        data.append({
            "transaction_id": transaction_id,
            "user_query": query,
            "item": item,
            "amount": amount,
            "prior_fraud_flag": prior_fraud_flag
        })

    df = pd.DataFrame(data)
    os.makedirs("backend/data", exist_ok=True)
    df.to_csv("transactions.csv", index=False)
    print(f"✅ Dataset generated at backend/data/transactions.csv with {n_rows} rows.")

if __name__ == "__main__":
    generate_dataset(600)
