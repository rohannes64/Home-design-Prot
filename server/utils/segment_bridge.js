/**
 * Segment Bridge — Node.js ↔ Python SegFormer bridge
 * 
 * Downloads a Cloudinary image to a temp file, spawns segment_room.py
 * with --json_only, parses the JSON output, and returns structured zone data.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const axios = require('axios');

const PYTHON_SCRIPT = path.join(__dirname, '..', 'segment_room.py');
const TIMEOUT_MS = 120_000; // 2 minutes (model loading on first run can be slow)
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

/**
 * Download image from URL to a temp file
 * @param {string} url - Image URL (typically Cloudinary)
 * @returns {Promise<string>} Path to downloaded temp file
 */
async function downloadToTemp(url) {
  const ext = path.extname(new URL(url).pathname) || '.jpg';
  const tmpPath = path.join(os.tmpdir(), `arteffects_seg_${Date.now()}${ext}`);

  const response = await axios({
    method: 'GET',
    url,
    responseType: 'stream',
    timeout: 30_000,
  });

  const writer = fs.createWriteStream(tmpPath);
  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', () => resolve(tmpPath));
    writer.on('error', (err) => {
      fs.unlink(tmpPath, () => {});
      reject(err);
    });
  });
}

/**
 * Run Python segmentation and return zone data
 * @param {string} imageUrl - Cloudinary URL of the room photo
 * @returns {Promise<{zones: Array, image_size: {width: number, height: number}}>}
 */
async function segmentImage(imageUrl) {
  let tmpPath = null;

  try {
    // 1. Download image to temp
    console.log('[SegBridge] Downloading image...');
    tmpPath = await downloadToTemp(imageUrl);
    console.log(`[SegBridge] Downloaded to: ${tmpPath}`);

    // 2. Send HTTP request to Python FastAPI microservice
    console.log(`[SegBridge] Sending request to Python API (${PYTHON_API_URL}/segment)...`);
    
    // We expect the Python server to already have the model loaded in RAM
    const response = await axios.post(`${PYTHON_API_URL}/segment`, {
      image_path: tmpPath
    }, {
      timeout: 30000 // Fast timeout, inference should take < 1s
    });

    const result = response.data;
    console.log(`[SegBridge] Detected ${result.zones?.length || 0} zones`);
    return result;

  } catch (err) {
    if (err.response) {
      // The Python server responded with a status code that falls out of the range of 2xx
      throw new Error(`Python API Error: ${err.response.data?.detail || err.message}`);
    } else if (err.code === 'ECONNREFUSED') {
      throw new Error('Python segmentation API is not running. Did you start it with uvicorn?');
    }
    throw err;
  } finally {
    // 3. Cleanup temp file
    if (tmpPath) {
      fs.unlink(tmpPath, (err) => {
        if (err) console.warn(`[SegBridge] Could not delete temp file: ${tmpPath}`);
      });
    }
  }
}

/**
 * Call local Python FastAPI for SD Generation
 * @param {string} imagePath - Local path to the temp image
 * @param {string} zone - The zone to replace
 * @param {string} texturePrompt - The texture prompt
 * @returns {Promise<object>} The result containing renderedUrl
 */
async function generateImage(imagePath, zone, texturePrompt, textureImagePath = null) {
  console.log(`[SegBridge] Sending request to Python API (${PYTHON_API_URL}/generate)...`);
  try {
    const response = await axios.post(`${PYTHON_API_URL}/generate`, {
      image_path: imagePath,
      zone: zone,
      texture_prompt: texturePrompt,
      texture_image_path: textureImagePath
    }, {
      timeout: 300000 // 5 minutes (allows for model loading on first run)
    });
    return response.data;
  } catch (err) {
    if (err.response) {
      throw new Error(`Python API Error: ${err.response.data?.detail || err.message}`);
    } else if (err.code === 'ECONNREFUSED') {
      throw new Error('Python generation API is not running. Did you start it with uvicorn?');
    }
    throw err;
  }
}

module.exports = { segmentImage, generateImage, downloadToTemp };
