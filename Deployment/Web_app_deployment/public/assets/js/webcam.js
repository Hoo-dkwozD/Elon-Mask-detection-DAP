let dict = {
    0: "with mask",
    1: "without mask"
};

//Holds the model object
let model;
let fm_model;
//Holds the context object of the canvas
let ctx;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");

function modelReady() {
    document.getElementById("status").innerHTML = 'Model Loaded';
};

function cameraReady() {
    document.getElementById("status").innerHTML = 'Camera Ready';
};

function sorry() {
    document.getElementById("status").innerHTML = 'Sorry! May need to wait for 3 minutes :<';
};

async function predict() {    
    //Draw the frames obtained from video stream on a canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const returnTensors = false;

    //Predict landmarks in hand (3D coordinates) in the frame of a video
    const predictions = await model.estimateFaces(video, returnTensors);
    if(predictions.length > 0) {
        for (let i = 0; i < predictions.length; i++) {
            var start = predictions[i].topLeft;
            var end = predictions[i].bottomRight;
            var size = [end[0] - start[0], end[1] - start[1]];
            var img = ctx.getImageData(start[0], start[1], size[0], size[1]);

            var new_img = tf.browser.fromPixels(img);
            var smallImg = tf.image.resizeBilinear(new_img, [224, 224]);

            var resized = tf.cast(smallImg, "int32");
            var tf4d_ = resized
                .expandDims(0)
                .toFloat()
                .div(tf.scalar(127))
                .sub(tf.scalar(1));

            //Perform the detection with your layer model:
            var pred = await fm_model.predict(tf4d_);

            const predVal = pred.argMax(1).dataSync()[0];

            new_img.dispose();
            smallImg.dispose();
            resized.dispose();

            ctx.beginPath();
            ctx.strokeStyle="white";
            ctx.lineWidth = "3";
            ctx.rect(start[0], start[1],size[0], size[1]);
            ctx.stroke();
            var text = dict[predVal];
            ctx.fillStyle = "white";
            ctx.fillText(text,start[0]+5,start[1]+20);
        }
    }
    setTimeout(predict, 300);
}

async function web_main() {
    //Load the Handpose model
    console.log('Start loading model');
    fm_model = await tf.loadLayersModel("https://raw.githubusercontent.com/Clairverbot/wakanda/main/model.json");
    model = await blazeface.load();
    modelReady();
    console.log("Models Loaded");
    //Start the video stream, assign it to the video element and play it
     //Start the video stream, assign it to the video element and play it
    if(navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({video: true})
            .then(stream => {
                //assign the video stream to the video element
                video.srcObject = stream;
                //start playing the video
                video.play();
                cameraReady();
                sorry();

            })
            .catch(e => {
                console.log("Error Occurred in getting the video stream");
            });
    }

    video.onloadedmetadata = () => {
        //Get the 2D graphics context from the canvas element
        ctx = canvas.getContext('2d');
        //Reset the point (0,0) to a given point
        // ctx.translate(canvas.width, 0);
        //Flip the context horizontally
        // ctx.scale(-1, 1);

        //Start the prediction indefinitely on the video stream
        setTimeout(predict, 300);
    };   
}

web_main();