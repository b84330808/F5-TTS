import asyncio
import logging
import socket
import time

import numpy as np
import pyaudio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def listen_to_F5TTS(text, server_ip="localhost", server_port=9998):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    await asyncio.get_event_loop().run_in_executor(None, client_socket.connect, (server_ip, int(server_port)))

    start_time = time.time()
    first_chunk_time = None

    async def play_audio_stream():
        nonlocal first_chunk_time
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True, frames_per_buffer=2048)

        try:
            # recv() returns whatever has arrived: a read can stop in the middle of a float32 sample,
            # or carry the trailing b"END" together with the last samples. Keep leftover bytes for the next read.
            buffer = b""
            while True:
                data = await asyncio.get_event_loop().run_in_executor(None, client_socket.recv, 8192)
                if not data:
                    break
                buffer += data

                # The audio is whole float32 samples, so b"END" can only start on a sample boundary.
                end_of_audio = len(buffer) % 4 == 3 and buffer.endswith(b"END")
                if end_of_audio:
                    buffer = buffer[:-3]

                n_bytes = len(buffer) - len(buffer) % 4
                if n_bytes:
                    audio_array = np.frombuffer(buffer[:n_bytes], dtype=np.float32)
                    stream.write(audio_array.tobytes())
                    buffer = buffer[n_bytes:]

                    if first_chunk_time is None:
                        first_chunk_time = time.time()

                if end_of_audio:
                    logger.info("End of audio received.")
                    break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        logger.info(f"Total time taken: {time.time() - start_time:.4f} seconds")

    try:
        data_to_send = f"{text}".encode("utf-8")
        await asyncio.get_event_loop().run_in_executor(None, client_socket.sendall, data_to_send)
        await play_audio_stream()

    except Exception as e:
        logger.error(f"Error in listen_to_F5TTS: {e}")

    finally:
        client_socket.close()


if __name__ == "__main__":
    text_to_send = "As a Reader assistant, I'm familiar with new technology. which are key to its improved performance in terms of both training speed and inference efficiency. Let's break down the components"

    asyncio.run(listen_to_F5TTS(text_to_send))
