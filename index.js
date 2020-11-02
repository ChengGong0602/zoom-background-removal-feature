async function onLoad() {

  const MODEL_URL = 'deeplabv3_mnv2_pascal_train_aug_web_model/model.json';
  // Model's input and output have width and height of 513.
  const TENSOR_EDGE = 513;
  const [model, stream] = await Promise.all([
      tf.loadGraphModel(MODEL_URL),
      navigator.mediaDevices.getUserMedia({
          video: {
              facingMode: 'user',
              frameRate: 30, width: 513, height: 513
          }
      })]);
  const video = document.createElement('video');
  video.autoplay = true;
  video.width = video.height = TENSOR_EDGE;
  const ctx = document.getElementById("canvas").getContext("2d");
  const videoCopy = ctx.canvas.cloneNode(false).getContext("2d");
  const maskContext = document.createElement('canvas').getContext("2d");
  maskContext.canvas.width = maskContext.canvas.height = TENSOR_EDGE;
  const img = maskContext.createImageData(TENSOR_EDGE, TENSOR_EDGE);
  let imgd = img.data;
  new Uint32Array(imgd.buffer).fill(0x00FFFF00);
  const render = () => {
      videoCopy.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
      const out = tf.tidy(() => {
          return model.execute({ 'ImageTensor': tf.browser.fromPixels(video).expandDims(0) });
      });
      const data = out.dataSync();
      for (let i = 0; i < data.length; i++) {
          imgd[i * 4 + 3] = data[i] == 15 ? 0 : 255;
      }
      maskContext.putImageData(img, 0, 0);
      ctx.drawImage(videoCopy.canvas, 0, 0);
      if (document.getElementById("show-background-toggle").checked)
          ctx.drawImage(maskContext.canvas, 0, 0, ctx.canvas.width,
              ctx.canvas.height);
      window.requestAnimationFrame(render);
  }
  video.oncanplay = render;
  video.srcObject = stream;
}