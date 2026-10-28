FROM node:22-alpine AS build
WORKDIR /app
COPY fastapi_react/frontend/package.json fastapi_react/frontend/package-lock.json fastapi_react/frontend/.npmrc ./
COPY fastapi_react/frontend/scripts ./scripts
RUN npm ci
COPY fastapi_react/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY fastapi_react/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
