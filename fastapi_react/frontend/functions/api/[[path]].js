export async function onRequest({ request, env }) {
  if (!env.F1_API_ORIGIN) {
    return new Response("API origin is not configured", { status: 503 });
  }

  let apiUrl;
  try {
    apiUrl = new URL(env.F1_API_ORIGIN);
  } catch {
    return new Response("API origin is invalid", { status: 500 });
  }

  if (
    apiUrl.protocol !== "https:" ||
    apiUrl.pathname !== "/" ||
    apiUrl.search ||
    apiUrl.hash ||
    apiUrl.username ||
    apiUrl.password
  ) {
    return new Response("API origin must be an HTTPS origin", { status: 500 });
  }

  const incomingUrl = new URL(request.url);
  apiUrl.pathname = incomingUrl.pathname;
  apiUrl.search = incomingUrl.search;

  return fetch(new Request(apiUrl, request));
}
