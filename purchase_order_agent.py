import requests

class PurchaseOrderAgent:
    def __init__(self, procurement_agent_url):
        self.procurement_agent_url = procurement_agent_url

    def check_and_reorder(self, item, current_stock, threshold):
        if current_stock < threshold:
            print(f"Stock for {item} is low. Consulting Procurement Agent...")
            payload = {
                "action": "find_supplier",
                "data": {"item": item, "required_quantity": threshold - current_stock},
            }
            response = requests.post(f"{self.procurement_agent_url}/process_request", json=payload).json()

            if response["status"] == "success":
                supplier = response["supplier"]
                print(f"Found supplier: {supplier}. Confirming order...")
                confirmation_payload = {
                    "action": "confirm_order",
                    "data": {"supplier": supplier, "item": item, "quantity": threshold - current_stock},
                }
                confirmation_response = requests.post(f"{self.procurement_agent_url}/process_request", json=confirmation_payload)
                return confirmation_response.json()
            else:
                return {"status": "error", "message": "Failed to find supplier."}
        else:
            return {"status": "ok", "message": "Stock level is sufficient."}

if __name__ == "__main__":
    # Interact with the Procurement Agent
    procurement_agent_url = "http://127.0.0.1:5000"
    purchase_agent = PurchaseOrderAgent(procurement_agent_url)
    response = purchase_agent.check_and_reorder("Laptop", current_stock=5, threshold=10)
    print("Final Response:", response)
