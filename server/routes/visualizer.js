const express = require('express');
const router = express.Router();
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const Render = require('../models/Render');
const Product = require('../models/Product');
const { protect, optionalAuth } = require('../middleware/auth');
const { uploadRoomPhoto, cloudinary } = require('../middleware/upload');

// POST /api/visualizer/upload — upload room photo
router.post('/upload', optionalAuth, uploadRoomPhoto, async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No photo uploaded' });

    res.json({
      photoUrl: req.file.path,
      publicId: req.file.filename,
      width: req.file.width,
      height: req.file.height
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/visualizer/segment — detect zones in uploaded photo
router.post('/segment', optionalAuth, async (req, res) => {
  try {
    const { photoUrl } = req.body;
    if (!photoUrl) return res.status(400).json({ error: 'Photo URL required' });

    // Call Replicate's segmentation model (SAM or similar)
    // Returns detected zones: floor, wall, ceiling, pillars, etc.
    const zones = await detectZones(photoUrl);
    res.json({ zones });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/visualizer/generate — main AI generation
router.post('/generate', optionalAuth, async (req, res) => {
  const startTime = Date.now();
  
  try {
    const { photoUrl, photoPublicId, appliedProducts, preset, userId } = req.body;
    
    if (!photoUrl) return res.status(400).json({ error: 'Photo URL required' });
    if (!appliedProducts?.length && !preset) {
      return res.status(400).json({ error: 'Select at least one product or preset' });
    }

    const hfKey = process.env.HF_API_KEY;
    const hasHF = hfKey && hfKey.trim() !== '';

    let renderedUrl, renderedPublicId, logAppliedProducts;

    if (hasHF && appliedProducts && appliedProducts.length > 0 && !preset) {
      console.log('Using Hugging Face SegFormer API + Hugging Face Inpainting API (No local Python)');
      
      // Resolve product information
      const ap = appliedProducts[0];
      const product = await Product.findById(ap.productId);
      if (!product) {
        return res.status(400).json({ error: 'Selected product not found' });
      }

      // 1. Get binary mask from Hugging Face SegFormer
      console.log(`Extracting mask from Hugging Face for zone: ${ap.zone}`);
      const maskBase64 = await getMaskFromHuggingFace(photoUrl, ap.zone);

      // 2. Build descriptive prompt for the stone texture replacement
      const finish = product.finish || 'polished';
      const categoryName = product.category ? product.category.replace('_', ' ') : 'stone';
      const inpaintPrompt = `highly detailed ${finish} ${product.name} ${categoryName} floor or wall tile, photorealistic interior render, 8k, warm lighting`;

      // 3. Send mask, base photo, and prompt to Hugging Face
      console.log('Calling Hugging Face Inpainting API...');
      const hfResult = await generateWithHuggingFaceInpainting(photoUrl, maskBase64, inpaintPrompt);
      renderedUrl = hfResult.renderedUrl;
      renderedPublicId = hfResult.renderedPublicId;

      logAppliedProducts = [{
        product: ap.productId,
        productName: product.name,
        zone: ap.zone,
        coverage: ap.coverage || 0
      }];

    } else {
      // Fallback: Original Stability AI / Replicate image generation
      const productDetails = await buildProductPrompt(appliedProducts, preset);
      const visualizerRes = await generateVisualization(
        photoUrl,
        productDetails.prompt,
        productDetails.negativePrompt
      );
      renderedUrl = visualizerRes.renderedUrl;
      renderedPublicId = visualizerRes.renderedPublicId;
      logAppliedProducts = productDetails.appliedProducts;
    }

    // Add watermark
    const watermarkedUrl = addWatermark(renderedUrl);
    
    // Save render to DB (requires auth for persistence, otherwise ephemeral)
    let render = null;
    const userIdToUse = req.user?._id || userId;
    
    if (userIdToUse) {
      render = await Render.create({
        user: userIdToUse,
        originalPhoto: { url: photoUrl, publicId: photoPublicId },
        renderedPhoto: { url: renderedUrl, publicId: renderedPublicId },
        appliedProducts: logAppliedProducts,
        preset: preset || null,
        generationStatus: 'completed',
        generationPrompt: preset || 'Hugging Face Inpainting Render',
        generationDuration: Date.now() - startTime,
        watermarkedUrl
      });
    }

    res.json({
      renderId: render?._id,
      originalUrl: photoUrl,
      renderedUrl: watermarkedUrl,
      hdUrl: renderedUrl,
      generationDuration: Date.now() - startTime,
      appliedProducts: logAppliedProducts
    });

  } catch (err) {
    console.error('Generation error:', err);
    res.status(500).json({ error: `Generation failed: ${err.message}` });
  }
});

// Helper: Build AI prompt from selected products
async function buildProductPrompt(appliedProducts = [], preset = null) {
  const productMap = {};
  const productIds = appliedProducts.map(p => p.productId).filter(Boolean);
  
  if (productIds.length) {
    const products = await Product.find({ _id: { $in: productIds } });
    products.forEach(p => { productMap[p._id.toString()] = p; });
  }

  const materialDescriptions = appliedProducts.map(ap => {
    const product = productMap[ap.productId];
    if (!product) return '';
    
    const finishDesc = {
      polished: 'highly polished with mirror-like reflections',
      honed: 'matte honed finish with subtle texture',
      brushed: 'brushed surface with linear texture',
      antique: 'aged antique finish with character',
      natural: 'natural split-face texture',
      flamed: 'rough flamed texture'
    }[product.finish] || 'polished';
    
    return `${ap.zone} covered with ${product.name} ${product.category.replace('_', ' ')} stone, ${finishDesc}, photorealistic material, proper perspective and lighting`;
  }).filter(Boolean);

  let presetDesc = '';
  if (preset === 'ionic_columns') {
    presetDesc = 'Add grand Ionic marble columns on sides, neoclassical architecture style';
  } else if (preset === 'cornice') {
    presetDesc = 'Add ornate white stone cornice moulding along ceiling edges';
  } else if (preset === 'wainscoting') {
    presetDesc = 'Add Gwalior stone wainscoting panels on lower wall section, traditional style';
  } else if (preset === 'full_neoclassical') {
    presetDesc = 'Transform to full neoclassical Indian interior with marble floors, white columns, ornate cornices';
  }

  const prompt = [
    'Interior design visualization, photorealistic render, 8K quality,',
    'Indian home interior, warm natural lighting,',
    ...materialDescriptions,
    presetDesc,
    'professional architectural photography, high detail, accurate material textures,',
    'maintain room proportions and existing furniture, realistic shadows and reflections'
  ].filter(Boolean).join(', ');

  const negativePrompt = [
    'distorted proportions, unrealistic materials, cartoon, illustration,',
    'blurry, low quality, dark, overexposed, missing walls, floating objects,',
    'changed room layout, removed doors or windows, wrong perspective'
  ].join(' ');

  return {
    prompt,
    negativePrompt,
    appliedProducts: appliedProducts.map(ap => ({
      product: ap.productId,
      productName: productMap[ap.productId]?.name || '',
      zone: ap.zone,
      coverage: ap.coverage || 0
    }))
  };
}

// Helper: Call Stability AI or Replicate for img2img
async function generateVisualization(imageUrl, prompt, negativePrompt) {
  const stabilityKey = process.env.STABILITY_API_KEY;
  const replicateToken = process.env.REPLICATE_API_TOKEN;
  
  const hasStability = stabilityKey && stabilityKey !== 'your_stability_ai_key' && stabilityKey.trim() !== '';
  const hasReplicate = replicateToken && replicateToken !== 'your_replicate_token' && replicateToken.trim() !== '';

  try {
    if (hasStability) {
      return await callStabilityAI(imageUrl, prompt, negativePrompt);
    } else if (hasReplicate) {
      return await callReplicate(imageUrl, prompt, negativePrompt);
    } else {
      // Demo mode: return original image with demo watermark
      console.warn('No AI API key configured, returning demo response');
      return { renderedUrl: imageUrl, renderedPublicId: null };
    }
  } catch (err) {
    throw new Error(`AI generation failed: ${err.message}`);
  }
}

async function callStabilityAI(imageUrl, prompt, negativePrompt) {
  // Download the init_image as an arraybuffer
  const imgResponse = await axios.get(imageUrl, { responseType: 'arraybuffer' });
  const blob = new Blob([imgResponse.data], { type: 'image/png' });

  const formData = new FormData();
  formData.append('init_image', blob, 'init_image.png');
  formData.append('image_strength', '0.35');
  formData.append('text_prompts[0][text]', prompt);
  formData.append('text_prompts[0][weight]', '1');
  formData.append('text_prompts[1][text]', negativePrompt);
  formData.append('text_prompts[1][weight]', '-1');
  formData.append('cfg_scale', '8');
  formData.append('steps', '50');
  formData.append('samples', '1');
  formData.append('style_preset', 'photographic');

  const response = await axios.post(
    'https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image',
    formData,
    {
      headers: {
        Authorization: `Bearer ${process.env.STABILITY_API_KEY}`,
        Accept: 'application/json'
      }
    }
  );

  const imageData = response.data.artifacts[0].base64;
  const uploaded = await cloudinary.uploader.upload(
    `data:image/png;base64,${imageData}`,
    { folder: 'arteffects/renders' }
  );

  return { renderedUrl: uploaded.secure_url, renderedPublicId: uploaded.public_id };
}

async function callReplicate(imageUrl, prompt, negativePrompt) {
  const axios_replicate = axios.create({
    headers: { Authorization: `Token ${process.env.REPLICATE_API_TOKEN}` }
  });

  // Start prediction
  const prediction = await axios_replicate.post('https://api.replicate.com/v1/predictions', {
    version: 'stability-ai/stable-diffusion-img2img:15a3689ee13b0d2616e98820eca31d4af4b51808d3547c1e34b6ce8bdea26dab',
    input: {
      image: imageUrl,
      prompt,
      negative_prompt: negativePrompt,
      prompt_strength: 0.7,
      num_inference_steps: 50
    }
  });

  // Poll for result
  let result = prediction.data;
  while (result.status !== 'succeeded' && result.status !== 'failed' && result.status !== 'canceled') {
    await new Promise(r => setTimeout(r, 1500));
    const poll = await axios_replicate.get(`https://api.replicate.com/v1/predictions/${result.id}`);
    result = poll.data;
  }

  if (result.status === 'failed' || result.status === 'canceled') {
    throw new Error(`Replicate generation ${result.status}`);
  }

  const renderedUrl = result.output[0];
  const uploaded = await cloudinary.uploader.upload(renderedUrl, {
    folder: 'arteffects/renders'
  });

  return { renderedUrl: uploaded.secure_url, renderedPublicId: uploaded.public_id };
}

// Helper: Add watermark via Cloudinary transformation
function addWatermark(imageUrl) {
  if (!imageUrl.includes('cloudinary.com')) return imageUrl;
  const watermarkText = encodeURIComponent(process.env.WATERMARK_TEXT || 'Arteffects');
  return imageUrl.replace('/upload/', `/upload/l_text:Arial_32_bold:${watermarkText},co_white,o_60,g_south_east,x_20,y_20/`);
}

// Helper: Extract specific zone mask as Base64 string using Hugging Face SegFormer API
async function getMaskFromHuggingFace(photoUrl, zoneName) {
  const hfKey = process.env.HF_API_KEY;
  if (!hfKey) {
    throw new Error('HF_API_KEY is not configured in the server environment (.env)');
  }

  // 1. Download original room image buffer
  let roomBuffer;
  try {
    const roomRes = await axios.get(photoUrl, { responseType: 'arraybuffer' });
    roomBuffer = Buffer.from(roomRes.data);
  } catch (err) {
    throw new Error(`Failed to download room image for segmentation: ${err.message}`);
  }

  // 2. Query Hugging Face SegFormer Model for segment maps
  try {
    const hfUrl = "https://api-inference.huggingface.co/models/nvidia/segformer-b2-finetuned-ade-512-512";
    const hfResponse = await axios.post(hfUrl, roomBuffer, {
      headers: {
        'Authorization': `Bearer ${hfKey}`,
        'Content-Type': 'image/jpeg'
      }
    });

    const segments = hfResponse.data;
    if (!Array.isArray(segments)) {
      throw new Error(`SegFormer returned invalid response format: ${JSON.stringify(segments)}`);
    }

    // Look for target zone name (e.g. wall/floor/ceiling)
    const targetSegment = segments.find(s => s.label.toLowerCase() === zoneName.toLowerCase());
    if (!targetSegment) {
      throw new Error(`Target zone '${zoneName}' was not detected in this room image by the AI model.`);
    }

    // Return the base64 mask string
    return targetSegment.mask;

  } catch (err) {
    console.error('Hugging Face SegFormer segmentation error:', err.response?.data ? err.response.data.toString() : err.message);
    throw new Error(`Hugging Face segmentation failed: ${err.message}`);
  }
}

// Helper: Call Hugging Face Stable Diffusion Inpainting API and upload output to Cloudinary
async function generateWithHuggingFaceInpainting(roomUrl, maskBase64, prompt) {
  const hfKey = process.env.HF_API_KEY;
  if (!hfKey) {
    throw new Error('HF_API_KEY is not configured in the server environment (.env)');
  }

  // 1. Download original room image and get base64
  let roomBase64;
  try {
    const roomRes = await axios.get(roomUrl, { responseType: 'arraybuffer' });
    roomBase64 = Buffer.from(roomRes.data).toString('base64');
  } catch (err) {
    throw new Error(`Failed to download room image for HF generation: ${err.message}`);
  }

  // 2. Call Hugging Face API
  try {
    const hfUrl = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting";
    const payload = {
      inputs: prompt,
      image: roomBase64,
      mask_image: maskBase64
    };

    const response = await axios.post(hfUrl, payload, {
      headers: {
        'Authorization': `Bearer ${hfKey}`,
        'Content-Type': 'application/json'
      },
      responseType: 'arraybuffer'
    });

    const finalBuffer = Buffer.from(response.data);

    // 3. Upload composited result to Cloudinary
    const uploadResult = await new Promise((resolve, reject) => {
      cloudinary.uploader.upload_stream(
        { folder: 'arteffects/renders' },
        (error, result) => {
          if (error) reject(error);
          else resolve(result);
        }
      ).end(finalBuffer);
    });

    return {
      renderedUrl: uploadResult.secure_url,
      renderedPublicId: uploadResult.public_id
    };

  } catch (err) {
    console.error('Hugging Face Inpainting API error:', err.response?.data ? err.response.data.toString() : err.message);
    throw new Error(`Hugging Face API call failed: ${err.message}`);
  }
}

module.exports = router;
