// Data Analysis App JavaScript

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Add smooth scrolling to all anchor links
    addSmoothScrolling();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Add loading states to buttons
    addLoadingStates();
    
    // Initialize file upload enhancements
    initializeFileUpload();
    
    // Add animations to cards
    addCardAnimations();
}

// Smooth scrolling for anchor links
function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Add loading states to form submissions
function addLoadingStates() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                
                // Re-enable button after 30 seconds as fallback
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }, 30000);
            }
        });
    });
}

// Enhanced file upload functionality
function initializeFileUpload() {
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        // Add drag and drop functionality
        const uploadArea = fileInput.parentElement;
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            uploadArea.classList.add('drag-over');
        }
        
        function unhighlight(e) {
            uploadArea.classList.remove('drag-over');
        }
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            updateFileLabel(files[0]);
        }
        
        // Update file label when file is selected
        fileInput.addEventListener('change', function() {
            updateFileLabel(this.files[0]);
        });
        
        function updateFileLabel(file) {
            if (file) {
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                const fileName = file.name;
                const fileInfo = document.createElement('div');
                fileInfo.className = 'mt-2 p-2 bg-light rounded';
                fileInfo.innerHTML = `
                    <i class="fas fa-file-csv text-success me-2"></i>
                    <strong>${fileName}</strong> (${fileSize} MB)
                `;
                
                // Remove existing file info
                const existingInfo = uploadArea.querySelector('.file-info');
                if (existingInfo) {
                    existingInfo.remove();
                }
                
                fileInfo.classList.add('file-info');
                uploadArea.appendChild(fileInfo);
            }
        }
    }
}

// Add staggered animations to cards
function addCardAnimations() {
    const cards = document.querySelectorAll('.card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('animate-in');
                }, index * 100);
            }
        });
    });
    
    cards.forEach(card => {
        observer.observe(card);
    });
}

// Utility functions for data analysis
const DataAnalysisUtils = {
    // Format numbers for display
    formatNumber: function(num, decimals = 2) {
        if (isNaN(num)) return 'N/A';
        return Number(num).toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    },
    
    // Create simple statistics summary
    calculateStats: function(numbers) {
        if (!Array.isArray(numbers) || numbers.length === 0) {
            return null;
        }
        
        const sorted = numbers.slice().sort((a, b) => a - b);
        const sum = numbers.reduce((a, b) => a + b, 0);
        const mean = sum / numbers.length;
        const median = sorted.length % 2 === 0
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];
        
        // Standard deviation
        const variance = numbers.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / numbers.length;
        const stdDev = Math.sqrt(variance);
        
        return {
            count: numbers.length,
            sum: sum,
            mean: mean,
            median: median,
            min: Math.min(...numbers),
            max: Math.max(...numbers),
            stdDev: stdDev,
            variance: variance
        };
    },
    
    // Generate color palette for charts
    generateColors: function(count) {
        const colors = [
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 99, 132, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
            'rgba(199, 199, 199, 0.8)',
            'rgba(83, 102, 255, 0.8)'
        ];
        
        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colors[i % colors.length]);
        }
        return result;
    },
    
    // Create histogram data from array of values
    createHistogram: function(values, bins = 10) {
        if (!values || values.length === 0) return null;
        
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min;
        const binSize = range / bins;
        
        const histogram = new Array(bins).fill(0);
        const labels = [];
        
        for (let i = 0; i < bins; i++) {
            labels.push((min + i * binSize).toFixed(2));
        }
        
        values.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
            histogram[binIndex]++;
        });
        
        return { labels, data: histogram };
    }
};

// Export for use in other scripts
window.DataAnalysisUtils = DataAnalysisUtils;

// Add CSS for drag and drop
const style = document.createElement('style');
style.textContent = `
    .drag-over {
        border-color: #667eea !important;
        background-color: #f0f4ff !important;
    }
    
    .animate-in {
        opacity: 1;
        transform: translateY(0);
        transition: all 0.6s ease;
    }
    
    .card {
        opacity: 0;
        transform: translateY(20px);
    }
    
    .card.animate-in {
        opacity: 1;
        transform: translateY(0);
    }
`;
document.head.appendChild(style);

// Console welcome message
console.log('%c🚀 Data Analysis App initialized successfully!', 
    'color: #667eea; font-size: 16px; font-weight: bold;');
console.log('%cBuilt with Java Spring Boot ☕', 
    'color: #f5576c; font-size: 12px;');
