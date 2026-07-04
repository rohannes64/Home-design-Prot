const { segmentImage } = require('./server/utils/segment_bridge');

// A sample public room photo URL (e.g. Wikimedia Commons) to test segmentation
const TEST_IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Living_room_interior.jpg/800px-Living_room_interior.jpg';

async function runTest() {
  console.log('Testing segment bridge...');
  try {
    const result = await segmentImage(TEST_IMAGE_URL);
    console.log('\n--- SUCCESS ---');
    console.log('Detected Zones:');
    result.zones.forEach(z => {
      console.log(`- ${z.name}: ${(z.coverage * 100).toFixed(1)}%`);
    });
  } catch (err) {
    console.error('\n--- ERROR ---');
    console.error(err);
  }
}

runTest();
