window.addEventListener("scroll", () => {
    document.querySelectorAll("section").forEach(section => {
        const position = section.getBoundingClientRect().top;
        const trigger = window.innerHeight / 1.2;

        if (position < trigger) {
            section.style.opacity = "1";
            section.style.transform = "translateY(0)";
        }
    });
});
