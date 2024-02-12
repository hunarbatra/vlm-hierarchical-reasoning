const puppeteer = require('puppeteer-extra');
const EventEmitter = require('events');

// add stealth plugin and use defaults (all evasion techniques) 
const StealthPlugin = require('puppeteer-extra-plugin-stealth') 
puppeteer.use(StealthPlugin()) 

const {executablePath} = require('puppeteer') 

class OpenAIChatController extends EventEmitter {
    constructor() {
        super();
        this.browser = null;
        this.page = null;
    }

    async initialize() {
        this.browser = await puppeteer.launch({
            headless: false, // Launch browser in non-headless mode so you can see what's happening
            userDataDir: "./user_data", // Persist user data so you can login
            executablePath: executablePath()
        });
        this.page = await this.browser.newPage();
        await this.page.exposeFunction('emitEndTurn', (data) => this.emit('end_turn', data));

        await this.preparePage();
    }

    async preparePage() {
        await this.page.goto('https://chat.openai.com/?model=gpt-4', {waituntil: 'load', timeout: 0});

        await this.page.waitForSelector('input[type="file"]', {waituntil: 'networkidle0'});
        await this.page.evaluate(() => {
            const {fetch: origFetch} = window;
            window.fetch = async (...args) => {
              const response = await origFetch(...args);
            
              if(args[0] === "https://chat.openai.com/backend-api/conversation") {
                console.log("intercepting conversation...");
                
                const { body } = response.clone();
                const raw = await new Response(body).text();
                const chunks = raw.split('\ndata: ');
                for(let chunk of chunks) {
                    chunk = chunk.trim();
                    if(chunk.startsWith('{')) {
                        console.log(chunk);
                        try { 
                            let msg = JSON.parse(chunk);
                            if(msg.message && msg.message.end_turn) {
                                console.log(msg.message.content.parts);
                                window.emitEndTurn(msg.message.content.parts.join(''));
                                break;
                            }
                        } catch( ex ) { }
                    }
                }
              }
            
              return response;
            };
        });
    }

    async typeIntoPrompt(text) {
        if (!this.page) {
            throw new Error('You need to initialize first');
        }
        await this.page.type('#prompt-textarea', text.split('\n').join(' '));
    }

    async clickSendButton() {
        if (!this.page) {
            throw new Error('You need to initialize first');
        }
        await this.page.waitForSelector('button[data-testid="send-button"]:not([disabled])', {waituntil: 'networkidle0'});
        await this.page.click('[data-testid="send-button"]');
    }

    async uploadImage(filePath) {
        if (!this.page) {
            throw new Error('You need to initialize first');
        }
        await this.page.reload();
        await this.preparePage();

        const input = await this.page.$('input[type="file"]');
        await input.uploadFile(filePath);
        // wait until upload is complete
        await this.page.waitForSelector('button[data-testid="send-button"]:not([disabled])');
    }

    async close() {
        if (this.browser) {
            await this.browser.close();
        }
    }

    async interactWithGradio(imagePath) {
        await this.page.goto('https://ee06329f28067bcfd9.gradio.live/', { waitUntil: 'networkidle0' });
    
        // Set the value of the first slider to 1.8 - now using 2.5 for SAM
        const firstSliderSelector = 'input[type="range"][id="range_id_0"]'; // Adjust the selector to match your first slider
        await this.page.waitForSelector(firstSliderSelector);
        await this.page.$eval(firstSliderSelector, (el, value) => el.value = value, 2.5);
        await this.page.$eval(firstSliderSelector, el => el.dispatchEvent(new Event('input')));
        await this.page.$eval(firstSliderSelector, el => el.dispatchEvent(new Event('change')));

        // Set the value of the second slider to 0.05 -> now using 0.1 for SAM
        const secondSliderSelector = 'input[type="range"][id="range_id_1"]'; // Adjust the selector to match your second slider
        await this.page.waitForSelector(secondSliderSelector);
        await this.page.$eval(secondSliderSelector, (el, value) => el.value = value, 0.1);
        await this.page.$eval(secondSliderSelector, el => el.dispatchEvent(new Event('input')));
        await this.page.$eval(secondSliderSelector, el => el.dispatchEvent(new Event('change')));

        // remove checkmark from "mark" - adding numbers has been confusing and hiding segments in the image
        await this.page.evaluate(() => {
            const labels = Array.from(document.querySelectorAll('label.svelte-1qxcj04')); // Get all labels with this class
            const markLabel = labels.find(label => label.textContent.includes('Mark')); // Find the label with the text 'Mark'
            if (markLabel) {
              const checkbox = markLabel.querySelector('input[type="checkbox"]'); // Find the checkbox inside that label
              checkbox.click(); // Click the checkbox
            }
          });

        await this.page.waitForTimeout(3000);

        // Upload the image
        const fileInputSelector = 'input[type="file"]'; // Adjust if necessary to the exact selector of the file input
        await this.page.waitForSelector(fileInputSelector);
        const input = await this.page.$(fileInputSelector);
        await input.uploadFile(imagePath);

        await this.page.waitForTimeout(5000); // This may need to be adjusted depending on the response time of the website

        // Click the 'Run' button to start the image processing
        const runButtonSelector = 'button#component-5'; // Make sure this selector is correct
        await this.page.waitForSelector(runButtonSelector, { visible: true });
        await this.page.click(runButtonSelector);
    
        // Wait for the base64 image string to be available
        let base64String = null;
        for (let attempt = 0; attempt < 10; attempt++) {
            await this.page.waitForTimeout(10000); // Wait for 10 seconds
            base64String = await this.page.evaluate(() => {
                const anchor = document.querySelector('div.icon-buttons a[href^="data:image/"]');
                return anchor ? anchor.href : null;
            });
            
            if (base64String) break; // If image is found, break the loop
        }
        
        return base64String;
    }    
}

module.exports = OpenAIChatController;