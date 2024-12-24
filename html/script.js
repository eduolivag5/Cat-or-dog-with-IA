var tamano = 400;
var video = document.getElementById("video");
var canvas = document.getElementById("canvas");
var otrocanvas = document.getElementById("otrocanvas");
var ctx = canvas.getContext("2d");
var currentStream = null;
var facingMode = "environment"; // Usa la cámara trasera por defecto

var modelo = null;

(async () => {
  console.log("Cargando modelo...");
  modelo = await tf.loadLayersModel("./modelIA/model.json");
  console.log("Modelo cargado");
})();

window.onload = function () {
  mostrarCamara();
};

// Función para permitir que se suelte el archivo en el área de drop
function allowDrop(event) {
  event.preventDefault();
}

// Función que maneja el archivo cuando se suelta en el área de drop
function handleDrop(event) {
  event.preventDefault();
  
  const file = event.dataTransfer.files[0];
  if (file) {
    document.getElementById("file-name").textContent = `Archivo seleccionado: ${file.name}`;
    processImage(file);
  }
}

// Función para procesar la imagen cargada
function processImage(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = new Image();
    img.onload = function() {
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}


// Función para borrar el canvas
function borrarImagen() {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Restablecer la clase de los resultados
  document.getElementById("dog-result").classList.remove("bg-blue-500");
  document.getElementById("cat-result").classList.remove("bg-pink-500");
}


// Evento para manejar la carga de imágenes desde archivo
document.getElementById("file-input").addEventListener("change", function(event) {
  var file = event.target.files[0];
  if (file) {
    var reader = new FileReader();
    reader.onload = function(e) {
      var img = new Image();
      img.onload = function() {
        canvas.width = tamano;
        canvas.height = tamano;
        ctx.drawImage(img, 0, 0, tamano, tamano);
        predecir();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});

function mostrarCamara() {
  var opciones = {
    audio: false,
    video: {
      facingMode: facingMode,
      width: tamano,
      height: tamano,
    },
  };

  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia(opciones)
      .then(function (stream) {
        currentStream = stream;
        video.srcObject = currentStream;
        procesarCamara();
        predecir();
      })
      .catch(function (err) {
        console.log(err);
      });
  }
}

function cambiarCamara() {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => {
      track.stop();
    });
  }

  facingMode = facingMode === "user" ? "environment" : "user";

  var opciones = {
    audio: false,
    video: {
      facingMode: facingMode,
      width: tamano,
      height: tamano,
    },
  };

  navigator.mediaDevices
    .getUserMedia(opciones)
    .then(function (stream) {
      currentStream = stream;
      video.srcObject = currentStream;
    })
    .catch(function (err) {
      console.log("Oops, hubo un error", err);
    });
}

function procesarCamara() {
  ctx.drawImage(video, 0, 0, tamano, tamano, 0, 0, tamano, tamano);
  setTimeout(procesarCamara, 20);
}

function predecir() {
  if (modelo != null) {
    resample_single(canvas, 100, 100, otrocanvas);

    // Hacer la predicción
    var ctx2 = otrocanvas.getContext("2d");
    var imgData = ctx2.getImageData(0, 0, 100, 100);

    var arr = [];
    var arr100 = [];

    for (var p = 0; p < imgData.data.length; p += 4) {
      var rojo = imgData.data[p] / 255;
      var verde = imgData.data[p + 1] / 255;
      var azul = imgData.data[p + 2] / 255;

      var gris = (rojo + verde + azul) / 3;

      arr100.push([gris]);
      if (arr100.length == 100) {
        arr.push(arr100);
        arr100 = [];
      }
    }

    arr = [arr];

    var tensor = tf.tensor4d(arr);
    var resultado = modelo.predict(tensor).dataSync();

    console.log(resultado)

    var respuesta;
    if (resultado <= 0.5) {
      respuesta = "Gato";
      document.getElementById("dog-result").classList.remove("bg-blue-500");
      document.getElementById("cat-result").classList.add("bg-pink-500");
    } else {
      respuesta = "Perro";
      document.getElementById("cat-result").classList.remove("bg-pink-500");
      document.getElementById("dog-result").classList.add("bg-blue-500");
    }
  }

  setTimeout(predecir, 150);
}

function resample_single(canvas, width, height, resize_canvas) {
  var width_source = canvas.width;
  var height_source = canvas.height;
  width = Math.round(width);
  height = Math.round(height);

  var ratio_w = width_source / width;
  var ratio_h = height_source / height;
  var ratio_w_half = Math.ceil(ratio_w / 2);
  var ratio_h_half = Math.ceil(ratio_h / 2);

  var ctx = canvas.getContext("2d");
  var ctx2 = resize_canvas.getContext("2d");
  var img = ctx.getImageData(0, 0, width_source, height_source);
  var img2 = ctx2.createImageData(width, height);
  var data = img.data;
  var data2 = img2.data;

  for (var j = 0; j < height; j++) {
    for (var i = 0; i < width; i++) {
      var x2 = (i + j * width) * 4;
      var weight = 0;
      var weights = 0;
      var weights_alpha = 0;
      var gx_r = 0;
      var gx_g = 0;
      var gx_b = 0;
      var gx_a = 0;
      var center_y = (j + 0.5) * ratio_h;
      var yy_start = Math.floor(j * ratio_h);
      var yy_stop = Math.ceil((j + 1) * ratio_h);
      for (var yy = yy_start; yy < yy_stop; yy++) {
        var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
        var center_x = (i + 0.5) * ratio_w;
        var w0 = dy * dy; // Pre-calc part of w
        var xx_start = Math.floor(i * ratio_w);
        var xx_stop = Math.ceil((i + 1) * ratio_w);
        for (var xx = xx_start; xx < xx_stop; xx++) {
          var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
          var w = Math.sqrt(w0 + dx * dx);
          if (w >= 1) {
            // Pixel too far
            continue;
          }
          // Hermite filter
          weight = 2 * w * w * w - 3 * w * w + 1;
          var pos_x = 4 * (xx + yy * width_source);
          // Alpha
          gx_a += weight * data[pos_x + 3];
          weights_alpha += weight;
          // Colors
          if (data[pos_x + 3] < 255) weight = weight * data[pos_x + 3] / 250;
          gx_r += weight * data[pos_x];
          gx_g += weight * data[pos_x + 1];
          gx_b += weight * data[pos_x + 2];
          weights += weight;
        }
      }
      data2[x2] = gx_r / weights;
      data2[x2 + 1] = gx_g / weights;
      data2[x2 + 2] = gx_b / weights;
      data2[x2 + 3] = gx_a / weights_alpha;
    }
  }
  ctx2.putImageData(img2, 0, 0);
}
