import { PlaygroundHost, type PlaygroundRequest } from '@sim/deep';
import type { Transport } from './client';

/** In-process stand-in for the worker (same host, asynchronous delivery). For tests only. */
export function inProcessTransport(): Transport & { crash(message: string): void } {
  const host = new PlaygroundHost();
  const transport: Transport & { crash(message: string): void } = {
    onmessage: null,
    onerror: null,
    postMessage(request: PlaygroundRequest) {
      const response = host.handle(request);
      queueMicrotask(() => transport.onmessage?.({ data: response }));
    },
    terminate() {},
    crash(message) {
      transport.onerror?.({ message });
    },
  };
  return transport;
}
