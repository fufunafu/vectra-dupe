import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { hashBytes } from '../server/static/sha256.mjs';

const bytes = text => new TextEncoder().encode(text);
const vectors = [
  ['', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'],
  ['abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'],
  ['abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
    '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1'],
  ['a'.repeat(1000000), 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0'],
];
for (const [input, expected] of vectors) {
  assert.equal(await hashBytes(bytes(input), null), expected, 'LAN fallback');
  assert.equal(await hashBytes(bytes(input), webcrypto.subtle), expected, 'Web Crypto path');
}
for (const length of [1, 55, 56, 63, 64, 65, 127, 128, 129, 2097153]) {
  const data = Uint8Array.from({ length }, (_, i) => (i * 29 + 7) % 256);
  const expected = createHash('sha256').update(data).digest('hex');
  assert.equal(await hashBytes(data.buffer, null), expected);
  const padded = new Uint8Array(length + 17);
  padded.set(data, 9);
  assert.equal(await hashBytes(padded.subarray(9, 9 + length), null), expected);
}
console.log('SHA-256 tests passed for LAN fallback, Web Crypto, padding boundaries, and large binary assets.');
