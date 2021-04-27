<template>
  <div id="app">
    <h1>AI-Powered Pokemon Types Classifier</h1>
    <p>
      This AI has learned how to recognize
      <a href="https://en.wikipedia.org/wiki/List_of_generation_I_Pok%C3%A9mon"
        >Pokemon from Generation 1</a
      >
      and guess its primary type by looking at it's image. There are a total of
      15 types. From our testing, the AI had an accuracy of 78.9%! Try below to
      see if it's right!
    </p>
    <h2>Pokemon Types</h2>
    <ul>
      <li>Bug</li>
      <li>Dragon</li>
      <li>Electric</li>
      <li>Fairy</li>
      <li>Fighting</li>
      <li>Fire</li>
      <li>Ghost</li>
      <li>Grass</li>
      <li>Ground</li>
      <li>Ice</li>
      <li>Normal</li>
      <li>Poison</li>
      <li>Psychic</li>
      <li>Rock</li>
      <li>Water</li>
    </ul>
    <p></p>
    <div>
      <h2>Upload image(s)</h2>
      <input type="file" multiple @change="onFileChange" accept="image/*" />
      <div v-for="(predictedImage, index) in predictedImages" :key="index">
        <img :src="predictedImage.dataUrl" @load="onImgLoad" />
        <div>Prediction: {{ predictedImage.prediction }}</div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, ref } from "vue";
import * as tf from "@tensorflow/tfjs";

interface predictedImage {
  dataUrl: string;
  prediction: string;
}

const pokemonTypes = [
  "Bug",
  "Dragon",
  "Electric",
  "Fairy",
  "Fighting",
  "Fire",
  "Ghost",
  "Grass",
  "Ground",
  "Ice",
  "Normal",
  "Poison",
  "Psychic",
  "Rock",
  "Water",
];

export default defineComponent({
  name: "App",

  setup() {
    let model: tf.GraphModel;
    const predictedImages = ref<predictedImage[]>([]);

    onMounted(async () => {
      try {
        model = await tf.loadGraphModel("trained_model/model.json");
      } catch (error) {
        console.log(error);
      }
    });

    const getImageURL = (file: File): Promise<string> => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (!e.target || !e.target.result) {
            return reject("Failed to load image");
          }
          resolve(e.target.result as string);
        };
        reader.readAsDataURL(file);
      });
    };

    const getImage = (url: string): Promise<tf.Tensor<tf.Rank>> => {
      return new Promise((resolve) => {
        const image = new Image();
        image.src = url;
        image.onload = () => {
          resolve(
            tf.browser
              .fromPixels(image) // returns a 3D tensor [width, height, channels]
              .resizeBilinear([32, 32]) // Model was trained on 32x32 images
              .expandDims(0) // model.predict() expects a 4D tensor [batch_size, width, height, channels]
          );
        };
      });
    };

    const getPrediction = (image: tf.Tensor<tf.Rank>) => {
      const result = model.predict(image) as tf.Tensor<tf.Rank>;
      return pokemonTypes[result.argMax(1).dataSync()[0]];
    };

    return {
      predictedImages,
      onFileChange: async (e: InputEvent) => {
        const inputElement = e.target as HTMLInputElement;
        if (!inputElement.files || !inputElement.files.length) {
          return;
        }

        try {
          const urls = await Promise.all(
            [...inputElement.files].map((file) => getImageURL(file))
          );
          const images = await Promise.all(urls.map((url) => getImage(url)));
          const predictions = await Promise.all(
            images.map((image) => getPrediction(image))
          );

          predictedImages.value = predictions.map((prediction, index) => {
            return {
              dataUrl: urls[index],
              prediction,
            };
          });
        } catch (error) {
          console.log(error);
        }
      },
    };
  },
});
</script>

<style>
html,
body {
  margin: 0;
  height: 100vh;
}
h1,
h2,
h3 {
  line-height: 1.2;
}

ul {
  text-align: left;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  height: 100vh;
  width: 100vw;
  text-align: center;
  color: #2c3e50;
  margin: 60px auto;
  max-width: 650px;
  line-height: 1.6;
  font-size: 18px;
  padding: 0 10px;
}

img {
  width: 256px;
  margin: auto;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
</style>
