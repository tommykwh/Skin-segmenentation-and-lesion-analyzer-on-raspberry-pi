import board
import neopixel
pixels = neopixel.NeoPixel(board.D21, 12)
# pixels[0] = (255, 0, 0)
pixels.fill((5, 5, 5))