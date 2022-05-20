/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import { setupStats } from './stats_panel';
import { Context } from './camera';
import { setupDatGui } from './option_panel';
import { STATE } from './params';
import { setBackendAndEnvFlags } from './util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
const statusElement = document.getElementById('status');

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: 500, height: 500 },
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
          STATE.model, { runtime, modelType: STATE.modelConfig.type });
      }
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
        posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
        posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, { modelType });
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    detector.dispose();

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
      1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  // FPS only counts the time it takes to finish estimatePoses.
  beginEstimatePosesStats();

  const poses = await detector.estimatePoses(
    camera.image,
    { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });

  console.log(poses)
  endEstimatePosesStats();

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
  }
}

async function checkUpdate() {
  await checkGuiUpdate();

  requestAnimationFrame(checkUpdate);
};

async function updateImage(event) {
  // Clear reference to any previous uploaded video.
  URL.revokeObjectURL(camera.image.src);
  const file = event.target.files[0];
  camera.image.src = URL.createObjectURL(file);
  camera.image.onload = function () {
    const imageWidth = camera.image.clientWidth;
    const imageHeight = camera.image.clientHeight;
    camera.image.width = imageWidth;
    camera.image.height = imageHeight;
    camera.canvas.width = imageWidth;
    camera.canvas.height = imageHeight;
  }

  statusElement.innerHTML = 'Image uploaded.';
}

async function runModel() {
  await renderResult();
}

async function run() {
  statusElement.innerHTML = 'Warming up model.';

  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  if (runtime === 'tfjs') {
    const warmUpTensor =
      tf.fill([camera.image.height, camera.image.width, 3], 0, 'float32');
    await detector.estimatePoses(
      warmUpTensor,
      { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });
    warmUpTensor.dispose();
    statusElement.innerHTML = 'Model is warmed up.';
  }

  await runModel();
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);
  stats = setupStats();
  detector = await createDetector();
  camera = new Context();

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  const runButton = document.getElementById('submit');
  runButton.onclick = run;

  const uploadButton = document.getElementById('imagefile');
  uploadButton.onchange = updateImage;

  checkUpdate();
};

app();
