// Workflow Interactive Script - Scroll-Based Navigation
let currentStep = 0;
const totalSteps = 9; // Updated to include step 0 (Gist of the HOW)

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    updateProgress();
    
    // Intersection Observer for scroll-based step detection
    const observerOptions = {
        threshold: 0.5, // Step is considered active when 50% visible
        rootMargin: '-20% 0px -20% 0px' // Trigger when step is in middle 60% of viewport
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const step = parseInt(entry.target.dataset.step);
                
                // Update current step
                if (step !== currentStep) {
                    currentStep = step;
                    updateProgress();
                }
                
                // Mark step as viewed
                entry.target.classList.add('viewed');
                entry.target.classList.add('active');
                
                // Remove active from other steps
                document.querySelectorAll('.workflow-step').forEach(s => {
                    if (parseInt(s.dataset.step) !== step) {
                        s.classList.remove('active');
                    }
                });
            } else {
                // Remove active class when step leaves viewport
                const step = parseInt(entry.target.dataset.step);
                if (step === currentStep) {
                    entry.target.classList.remove('active');
                }
            }
        });
    }, observerOptions);
    
    // Observe all workflow steps
    document.querySelectorAll('.workflow-step').forEach(step => {
        observer.observe(step);
    });
    
    // Initial: Mark first step as active
    const firstStep = document.querySelector('[data-step="0"]');
    if (firstStep) {
        firstStep.classList.add('active');
        firstStep.classList.add('viewed');
    }
});

function updateProgress() {
    const progress = ((currentStep + 1) / totalSteps) * 100;
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = progress + '%';
    }
}

// Smooth scroll on step header click
document.querySelectorAll('.step-header').forEach(header => {
    header.style.cursor = 'pointer';
    header.addEventListener('click', function() {
        const stepElement = this.closest('.workflow-step');
        if (stepElement) {
            stepElement.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        }
    });
});
