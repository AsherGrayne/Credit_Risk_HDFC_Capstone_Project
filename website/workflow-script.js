// Workflow Interactive Script
let currentStep = 1;
const totalSteps = 8;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    updateProgress();
    showStep(currentStep);
    updateNavigation();
    
    // Scroll to top
    window.scrollTo(0, 0);
    
    // Intersection Observer for auto-activation
    const observerOptions = {
        threshold: 0.3,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const step = parseInt(entry.target.dataset.step);
                if (step <= currentStep) {
                    entry.target.classList.add('viewed');
                }
                if (step === currentStep) {
                    entry.target.classList.add('active');
                }
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.workflow-step').forEach(step => {
        observer.observe(step);
    });
});

function showStep(step) {
    // Remove active class from all steps
    document.querySelectorAll('.workflow-step').forEach(s => {
        s.classList.remove('active');
    });
    
    // Add active class to current step
    const currentStepElement = document.querySelector(`[data-step="${step}"]`);
    if (currentStepElement) {
        currentStepElement.classList.add('active');
        currentStepElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    // Mark previous steps as viewed
    for (let i = 1; i < step; i++) {
        const prevStep = document.querySelector(`[data-step="${i}"]`);
        if (prevStep) {
            prevStep.classList.add('viewed');
        }
    }
    
    updateProgress();
    updateNavigation();
}

function navigateStep(direction) {
    const newStep = currentStep + direction;
    
    if (newStep >= 1 && newStep <= totalSteps) {
        currentStep = newStep;
        showStep(currentStep);
    }
}

function updateProgress() {
    const progress = (currentStep / totalSteps) * 100;
    document.getElementById('progressBar').style.width = progress + '%';
}

function updateNavigation() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    prevBtn.disabled = currentStep === 1;
    nextBtn.disabled = currentStep === totalSteps;
    
    if (currentStep === totalSteps) {
        nextBtn.textContent = 'Complete ✓';
    } else {
        nextBtn.textContent = 'Next →';
    }
}

// Keyboard navigation
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        navigateStep(1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        navigateStep(-1);
    }
});

// Smooth scroll on step click
document.querySelectorAll('.step-header').forEach(header => {
    header.style.cursor = 'pointer';
    header.addEventListener('click', function() {
        const step = parseInt(this.closest('.workflow-step').dataset.step);
        currentStep = step;
        showStep(currentStep);
    });
});

