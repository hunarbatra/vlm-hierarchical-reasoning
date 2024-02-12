const express = require('express');
const bodyParser = require('body-parser');

const puppeteer = require('puppeteer');

const fs = require('fs').promises;
const path = require('path');

const OpenAIChatController = require('./chatgpt');
const { getImageAsBase64 } = require('./utils')

const app = express();
app.use(bodyParser.json({ limit: '50mb' }));

app.post('/gpt-4v', async (req, res) => {
    let controller;
    try {
        const { image, prompt } = req.body;

        if (!prompt) {
            return res.status(400).send('Prompt text is required.');
        }

        controller = new OpenAIChatController();
        await controller.initialize();

        if (image) { 
            const base64Data = image.replace(/^data:image\/png;base64,/, "");
            await fs.writeFile("out.png", base64Data, 'base64'); 
            await controller.uploadImage('out.png');
            await controller.page.waitForTimeout(3500);
        }

        await controller.typeIntoPrompt(prompt);
        await controller.page.waitForTimeout(4500);
        await controller.clickSendButton();
        await controller.page.waitForTimeout(4500);

        let checkIntervalId;
        let timeoutId;

        const output = await new Promise((resolve, reject) => {
            // Function to check if the response has been received
            const checkForResponse = async () => {
                // Check for the response condition here
                // For example, you might check if a certain element contains the expected response text
                const response = await controller.page.evaluate(() => {
                    // Replace 'responseSelector' with the appropriate selector for the response element
                    const responseEl = document.querySelector('responseSelector');
                    return responseEl ? responseEl.textContent : null;
                });

                // If the response is found, clear intervals and timeouts, resolve the promise
                if (response) {
                    clearInterval(checkIntervalId);
                    clearTimeout(timeoutId);
                    resolve(response);
                }
            };

            // Poll every 5 seconds for the output
            checkIntervalId = setInterval(() => {
                checkForResponse();
            }, 1000);

            // Set a timeout for 90 seconds in case no response is received
            timeoutId = setTimeout(() => {
                clearInterval(checkIntervalId); // Clear the interval checking for response
                reject(new Error('Response timed out')); // Reject the promise after 90 seconds
            }, 120000);

            // You should still attach to the 'end_turn' event in case it comes before the polling detects it
            controller.on('end_turn', (response) => {
                clearInterval(checkIntervalId);
                clearTimeout(timeoutId);
                resolve(response);
            });

        });

        // After the promise is resolved or rejected, clear the intervals and timeouts
        clearInterval(checkIntervalId);
        clearTimeout(timeoutId);

        res.json({ output });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).send(error);
    } finally {
        if (controller) {
            await controller.close(); // close pupeteer session
        }
    }
});

app.post('/dall-e3', async (req, res) => {
    let controller;
    try {
        const { prompt } = req.body;
        if (!prompt) {
            return res.status(400).send('Prompt text is required.');
        }

        controller = new OpenAIChatController();
        await controller.initialize();

        await controller.page.goto('https://chat.openai.com/?model=gpt-4-dalle', {waituntil: 'networkidle0'});

        await controller.page.waitForTimeout(3500);

        await controller.typeIntoPrompt(prompt);
        await controller.clickSendButton();

        await controller.page.waitForFunction(
            () => {
              // Check for the presence of images with a specific attribute or identifier
              const images = document.querySelectorAll('img[alt="Generated by DALL·E"]');
              return images.length > 0;
            },
            { polling: 'raf', timeout: 95000} 
        );

        await controller.page.waitForTimeout(2000);
        
        const imageUrls = await controller.page.evaluate(() => {
            const images = Array.from(document.querySelectorAll('img[alt="Generated by DALL·E"]'));
            console.log('length of images arr')
            console.log(images.length)
            return images.map(img => img.src);
        });

        const imagesBase64 = [];
        for (let url of imageUrls) {
            const imageBase64 = await getImageAsBase64(url);
            imagesBase64.push(imageBase64);
        }

        res.json({ images: imagesBase64 });

    } catch (error) {
        console.error('Error:', error);
        res.status(500).send(error);
    } finally {
        if (controller) {
            await controller.close();
        }
    }
});

app.post('/semantic-sam', async (req, res) => {
    let controller;
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).send('Base64 image string is required.');
        }

        // Convert base64 to an image file
        const imagePath = 'image.png'; // specify the path to save the image
        const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, "");
        await fs.writeFile(imagePath, base64Data, 'base64');

        controller = new OpenAIChatController();
        await controller.initialize();

        const base64String = await controller.interactWithGradio(imagePath);

        if (base64String) {
            res.json({ image: base64String });
        } else {
            res.status(408).send('Request Timeout: Unable to retrieve the image.');
        }

    } catch (error) {
        console.error('Error:', error);
        res.status(500).send(error.message);
    } finally {
        if (controller) {
            await controller.close();
        }
    }
});

app.post('/sdxl', async (req, res) => {
    let controller;
    try {
        const { prompt } = req.body;

        if (!prompt) {
            return res.status(400).send('Prompt text is required.');
        }

        controller = new OpenAIChatController();
        await controller.initialize();

        await controller.page.goto('https://huggingface.co/spaces/google/sdxl', {waituntil: 'networkidle0'});

        // await controller.page.waitForSelector('input.scroll-hide.svelte-1f354aw', {waituntil: 'networkidle0'});
        // await controller.page.type('input.scroll-hide.svelte-1f354aw', prompt);
        await controller.page.waitForTimeout(4000); // Wait for 4 seconds after clicking the button

        // await controller.page.click('button#gen-button');
        await controller.page.evaluate(() => {
            document.querySelector('button#gen-button').click();
          });
        await controller.page.waitForTimeout(5000); // Wait for 4 seconds after clicking the button

        // Wait additional time for images to load or 10 seconds whichever comes first
        await controller.page.waitForFunction(() => {
            // Check if the images are loaded
            const thumbnails = document.querySelectorAll('button.thumbnail-item');
            return thumbnails.length > 0;
        }, { polling: 'raf', timeout: 20000 });

        const imagesBase64 = await controller.page.evaluate(() => {
            const base64Strings = [];
            for (let i = 1; i <= 4; i++) { // Assuming there are 4 thumbnails
                const imageElement = document.querySelector(`button.thumbnail-item[aria-label="Thumbnail ${i} of 4"] img`);
                if (imageElement) {
                    base64Strings.push(imageElement.src);
                }
            }
            return base64Strings;
        });

        res.json({ images: imagesBase64 });

    } catch (error) {
        console.error('Error:', error);
        res.status(500).send(error);
    } finally {
        if (controller) {
            await controller.close();
        }
    }
});


const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}.`);
});