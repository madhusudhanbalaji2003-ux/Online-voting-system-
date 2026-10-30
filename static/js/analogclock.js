function initAnalogClock(containerId) {
    const clock = document.getElementById(containerId);
    if (!clock) return;

    const seconds = clock.querySelector('.seconds');
    const minutes = clock.querySelector('.minutes');
    const hour = clock.querySelector('.hour');
    const minuteText = clock.querySelector('.minute');

    // Create 60 spikes for seconds and minutes
    for(let s = 0; s < 60; s++){
        let mSpikeEl = document.createElement('i');
        let sSpikeEl = document.createElement('i');
        mSpikeEl.className = 'spike';
        sSpikeEl.className = 'spike';
        mSpikeEl.style.setProperty('--rotate', `${6 * s}deg`);
        sSpikeEl.style.setProperty('--rotate', `${6 * s}deg`);
        mSpikeEl.setAttribute('data-i', s);
        sSpikeEl.setAttribute('data-i', s);
        seconds.appendChild(sSpikeEl);
        minutes.appendChild(mSpikeEl);
    }

    function pad2(n) { return n < 10 ? '0' + n : String(n); }

    function getTime() {
        let date = new Date(),
        s  = date.getSeconds(),
        m  = date.getMinutes(),
        h  = date.getHours();

        // Display hour in 12h format
        let displayH = h % 12 || 12;

        hour.textContent = pad2(displayH);
        minuteText.textContent = pad2(m);
      
        minutes.style.setProperty('--dRotate', `${6 * m}deg`);

        if(s === 0){
            seconds.classList.add('stop-anim');
        } else {
            seconds.classList.remove('stop-anim');
        }
        if(m === 0){
            minutes.classList.add('stop-anim');
        } else {
            minutes.classList.remove('stop-anim');
        }
        
        seconds.style.setProperty('--dRotate', `${6 * s}deg`);
    }

    getTime();
    setInterval(getTime, 1000);
}

// Auto-initialize if an element with ID 'home-analog-clock' is present
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initAnalogClock('home-analog-clock');
    });
} else {
    initAnalogClock('home-analog-clock');
}
