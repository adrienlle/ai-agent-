import uvicorn

if __name__ == "__main__":
    uvicorn.run("web_server:app", host="0.0.0.0", reload=True)
