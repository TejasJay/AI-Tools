from mem0 import Memory
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

config = {
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-large", "embedding_dims": 1536},
    },
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://memgraph:7687",
            "username": "dummy",  # required even if unused
            "password": "dummy"
        }
    }
}

memory = Memory.from_config(config_dict=config)

def chat_bot(message: str, user_id: str = "default"):
    results = memory.search(query=message, user_id=user_id)
    memories_str = "\n".join(f"- {m['memory']}" for m in results["results"])

    messages = [
        {"role": "system", "content": f"You are a helpful AI. Here is relevant memory:\n{memories_str}"},
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    reply = response.choices[0].message.content

    memory.add([
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ], user_id=user_id)

    return reply

def main():
    print("🧠 Mem0 + Memgraph AI ready!")
    while True:
        try:
            msg = input("You: ").strip()
            if msg.lower() in {"exit", "quit"}:
                break
            print("AI:", chat_bot(msg))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
