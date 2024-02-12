const axios = require('axios');

async function getImageAsBase64(url) {
    try {
        // Get the image data as a Buffer
        const response = await axios.get(url, { responseType: 'arraybuffer' });

        // Convert the Buffer to a base64 string
        const base64Image = Buffer.from(response.data, 'binary').toString('base64');

        return base64Image;
    } catch (error) {
        console.error('Error fetching image:', error);
        throw error; 
    }
}

module.exports = {
    getImageAsBase64,
}