import asyncio
import json

model = 'jens_2l128bi'
words = 30
temp = 0.5

modelpath = F"/mnt/data/textgenrnn/{model}"
weights = F"{modelpath}/{model}_weights.hdf5"
vocab = F"{modelpath}/{model}_vocab.json"
config = F"{modelpath}/{model}_config.json"

async def run():
    resp = await socket_cmd({"cmd": "be", "weights": weights, "vocab": vocab, "config": config, "words": words, "temp": temp})
    (generated, num_words) = resp['result']
    print(f'Generated {num_words} in {resp["time"]:.2f}s\n{generated}')

async def socket_cmd(cmd):
    reader, writer = await asyncio.open_connection('10.1.0.2', 7762)
    writer.write(json.dumps(cmd).encode('utf-8'))
    await writer.drain()
    data = await reader.read()
    resp = json.loads(data.decode('utf-8'))
    writer.close()
    await writer.wait_closed()
    return resp

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
