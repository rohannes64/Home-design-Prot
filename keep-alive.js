const https = require('https');
const http = require('http');

// Replace this with your actual deployed backend URL
// It's best to hit a lightweight public endpoint like the products list
const BACKEND_URL = process.env.PING_URL || 'https://your-api-domain.com/api/products';

// Ping every 10 minutes (600,000 milliseconds)
// Render/Heroku typically sleep after 15-30 mins of inactivity
const PING_INTERVAL = 10 * 60 * 1000; 

function pingBackend() {
  const lib = BACKEND_URL.startsWith('https') ? https : http;
  
  console.log(`[${new Date().toISOString()}] Pinging backend to keep it awake...`);
  
  lib.get(BACKEND_URL, (res) => {
    console.log(`[${new Date().toISOString()}] Backend responded with status: ${res.statusCode}`);
  }).on('error', (err) => {
    console.error(`[${new Date().toISOString()}] Error pinging backend:`, err.message);
  });
}

// Run the first ping immediately
pingBackend();

// Schedule the recurring pings
setInterval(pingBackend, PING_INTERVAL);

console.log(`Keep-alive bot started.`);
console.log(`Target: ${BACKEND_URL}`);
console.log(`Interval: Every ${PING_INTERVAL / 60000} minutes.`);
console.log(`Press Ctrl+C to stop.`);
