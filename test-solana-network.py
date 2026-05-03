from solana.rpc.api import Client

client = Client("http://127.0.0.1:8899")
print(client.is_connected())
print(client.get_version())