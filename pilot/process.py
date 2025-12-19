import matplotlib.pyplot as plt
from utils import extract_rgb_signals_rect, get_spectrum, plot_rgb_signals

video_path = r'C:\Users\Sophia\Documents\rPPG\initial_tests\videos\60s.avi'
fps = 25
# roi = (240, 160, 440, 270)
# video_path = r'C:\Users\Sophia\Documents\UTI-11-12-2025\L7-11-12-2025-16-23.avi'
# video_path = r'C:\Users\Sophia\Documents\UTI-11-12-2025\L7-11-12-2025-16-26.avi'
roi = (1200, 500, 1600, 800)
# video_path = r'C:\Users\Sophia\Documents\UTI-11-12-2025\L9-11-12-2025-16-04.avi'
# roi = (180, 580, 350, 680) #(300, 330, 580, 440)
# video_path = r"C:\Users\Sophia\Documents\UTI-11-12-2025\L8-11-12-2025-16-40.avi"
# roi = (840, 140, 1000, 220)


df = extract_rgb_signals_rect(video_path, roi)
plot_rgb_signals(df, fps)
r, g, b, r_fft, g_fft, b_fft, freqs = get_spectrum(df, 0.75, 10, filter_order=4, fps=25)

# This will now display all figures created by the functions above at once.
plt.show()