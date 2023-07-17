var el = x => document.getElementById(x);

// For Image
var dataImage = localStorage.getItem('imgData');
var modelSelected = localStorage.getItem("modelSelected");
displayedImage = el('saved_image');
displayedImage.src = "data:image/jpeg;base64," + dataImage;

// let url = "http://localhost:8000/";
url = "/api/predict-breed/";

headers = {
    "Content-Type": "application/json"
}
data = {
    method: "POST",
    headers: headers,
    body: JSON.stringify({
        "image_bytes": dataImage,
        "task": modelSelected,
        // "task": "det",
        "top_n": 4
    })
}
console.log("Sending request to:", url);
let result_=0;
// var canvas;
function renderPlots(boxCoordinates, classes, scores) {
    var canvas = document.createElement('canvas');
    w = displayedImage.width;
    h = displayedImage.height;
    canvas.width = w;
    canvas.height = h;

    var ctx = canvas.getContext('2d');
    ctx.drawImage(displayedImage, 0, 0);
    for (var i = 0; i < boxCoordinates.length; i++) {
        var coords = boxCoordinates[i];
        var [x1, y1, x2, y2] = coords;

        // Calculate the rectangle position and size
        var rectX = x1 * w;
        var rectY = y1 * h;
        var rectWidth = (x2 - x1) * w;
        var rectHeight = (y2 - y1) * h;

        // Set the rectangle style
        boxLabel = classes[i];
        classConfidence = scores[i];

        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        // Draw the rectangle
        ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);


        ctx.font = '12px Arial';
        ctx.fillStyle = 'red';
        test = `${boxLabel}: ${(classConfidence).toFixed(2)}`;
        ctx.fillText(test, rectX + 5, rectY + 20);


    }
    displayedImage.src = canvas.toDataURL();
}
fetch(url, data).then(response => response.json()).then(data => {
    console.log("data", data);
    result_ = data;
    // el('result_txt').innerHTML = JSON.stringify(data);
    // iterate result_ Object
    let resultText = "";
    if(modelSelected == "clf") {
        for (let key in result_) {
            let chance = (result_[key]*100).toFixed(1);
            if(chance > 5) {
            resultText += `${key}: ${chance} %<br>`;
            }
            if (resultText.length < 1) {
                resultText = "No Dog Detected";
            }
        }
    }
    else if(modelSelected == "det") {
        cls = result_["cls"];
        numDetected = cls.length;
        if (numDetected > 0) {
            resultText = `Detected ${numDetected} dogs<br> ${cls.join("<br>")}`;
            boxes = result_["xyxyn"];
            classes = result_["cls"];
            scores = result_["conf"];
            renderPlots(boxes, classes, scores);
        }
        else {
            resultText = "No Dog Detected";
        }
    }
    el('result_txt').innerHTML = resultText;
}
)
