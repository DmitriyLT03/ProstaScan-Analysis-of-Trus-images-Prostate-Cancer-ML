import requests

with open("image.jpg", "rb") as f:
    response = requests.post("http://localhost:3000/api/process_image", files={"image": f})
with open("result.jpeg", "wb") as f:
    f.write(response.content)