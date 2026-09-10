const jsonHeaders = { "Content-Type": "application/json" };

async function parse(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return body;
}

export const api = {
  get: async (url) => parse(await fetch(url)),
  post: async (url, body) => parse(await fetch(url, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(body)
  }))
};

export function downloadUrl(path) {
  return `/api/raw/download?path=${encodeURIComponent(path)}`;
}
