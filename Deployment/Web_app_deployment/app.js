// Include required packages
const express = require('express');
var compression = require('compression');
var helmet = require('helmet');
const app = express();

// Add compression middleware
app.use(compression());

// Add helmet middleware
app.use(helmet());
// Specify CSP directives to access external JS
app.use(helmet.contentSecurityPolicy({
    directives: {
        "default-src": ["'self'", "giphy.com", "https://raw.githubusercontent.com/Clairverbot/wakanda/main"],
        "script-src": ["'self'", "cdn.jsdelivr.net"],
    },
}));

// Last middleware to serve static files
app.use(express.static('public'));

// Listen on the appropriate port
app.listen(process.env.PORT || 3000);

module.exports = app;