() => {
  // HF's attribution avatar is injected after the application loads.
  const labelAvatar = () => {
    const images = document.querySelectorAll('img[src^="https://huggingface.co/api/users/kiselyovd/avatar"]');
    for (const image of images) {
      if (!image.hasAttribute('alt')) {
        const ownerLink = image.closest('a');
        image.alt = ownerLink?.textContent?.includes('kiselyovd') ? '' : 'kiselyovd';
      }
    }
    return images.length > 0;
  };
  if (!labelAvatar()) {
    const observer = new MutationObserver(() => {
      if (labelAvatar()) observer.disconnect();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
}
