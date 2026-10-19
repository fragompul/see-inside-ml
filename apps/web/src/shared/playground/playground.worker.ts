/// <reference lib="webworker" />
import { PlaygroundHost, type PlaygroundRequest } from '@sim/deep';

// All training runs here so the UI thread never blocks. The host is transport-agnostic and
// unit-tested in the deep package; this file only wires it to postMessage.
const host = new PlaygroundHost();
const scope = self as unknown as DedicatedWorkerGlobalScope;

scope.onmessage = (event: MessageEvent<PlaygroundRequest>) => {
  scope.postMessage(host.handle(event.data));
};
