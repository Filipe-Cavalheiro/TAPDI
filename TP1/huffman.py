
#from multiprocessing import Queue
from queue import PriorityQueue
from collections import Counter
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def huffman(symbol_list):
    """
    This function generates the huffman tree for the given input.
    The input is a list of "symbols".
    """
    # figure out the frequency of each symbol
    counts = Counter(symbol_list).most_common()

    total = len(symbol_list)
    if len(counts) < 2:
        # 0 or 1 unique symbols, so no sense in performing huffman coding
        return

    queue = PriorityQueue()
    for (val,count) in counts:
        queue.put((count, val))

    # Create the huffman tree
    largest_node_count = 0
    while total != largest_node_count:
        node1 = queue.get(False)
        node2 = queue.get(False)

        new_count = node1[0] + node2[0]
        largest_node_count = new_count if new_count > largest_node_count else largest_node_count
        queue.put((new_count, (node1,node2)))
    huffman_tree_root = queue.get(False)

    # generate the symbol to huffman code mapping
    lookup_table = huffman_tree_to_table(huffman_tree_root, "", {})
    return lookup_table

def huffman_tree_to_table(root, prefix, lookup_table):
    """Converts the Huffman tree rooted at "root" to a lookup table"""
    if type(root[1]) != tuple:
        # leaf node
        lookup_table[root[1]] = prefix
    else:
        huffman_tree_to_table(root[1][0], prefix + "0", lookup_table)
        huffman_tree_to_table(root[1][1], prefix + "1", lookup_table)

    return lookup_table

def text_to_huffman_code(input_text):
    """Helper function to convert an input string into its huffman symbol table"""
    return huffman([c for c in input_text])

def compress_img(huffman_dict: dict, img: list) -> str:
    #Ex 3.4

    final_bit_str: str = f"{len(huffman_dict.keys()):08b}"
    code_bits_dimension = int(len(bin(max(list(map(lambda x: len(x), huffman_dict.values()))))) - 2)
    final_bit_str += f"{code_bits_dimension:08b}" 

    sorted_dict = dict(sorted(huffman_dict.items(), key = lambda x: x[0]))

    for k, i in sorted_dict.items():
        final_bit_str += f"{k:08b}{int(i, 2):0{code_bits_dimension}b}"

    for i in img:
        final_bit_str += f"{int(sorted_dict.get(i), 2):0{code_bits_dimension}b}"

    return final_bit_str

def entropy(img):
    #Ex 7

    total_entropy = np.array([])

    for i in range(3):
        histogram = np.array(cv.calcHist([img],[i],None,[256],[0,256]))
        total_inst = np.sum(histogram)

        histogram = histogram/total_inst

        histogram = histogram[histogram != 0]

        total_entropy = np.append(total_entropy, np.sum(-histogram*np.log2(histogram)))

    return total_entropy

def main():
    img = cv.imread("aula1.bmp", 0) 
    plt.imshow(img, cmap="gray")
    plt.show()

    #Ex 1
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    
    plt.plot(hist), plt.xlim([0, 256])
    plt.show()

    #Ex 2
    flat_image_bin = ''.join(list(map(lambda x: f"{x:08b}", flat_img)))
    print(f"Uncompressed image: {flat_image_bin}")
    print(f"Uncompressed image bit length {len(flat_image_bin)}")

    #Ex 3.2
    flat_img = img.flatten()
    huffman_dict = huffman(flat_img)
    print(huffman_dict)

    #Ex 3.4
    compressed = compress_img(huffman_dict, flat_img)
    print(f"Compressed image: {compressed}")
    print(f"Compressed image bit length: {len(compressed)}")

    #Ex 4
    print(f"Compression ratio (Uncompressed/Compressed): {len(flat_image_bin)/len(compressed)}")


    #Ex 7
    img1 = cv.imread("Picture1.png")
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

    plt.imshow(img1)
    plt.show()
    
    entropy_per_channel = entropy(img1)

    print(f"Blue channel entropy: {entropy_per_channel[0]}")
    print(f"Green channel entropy: {entropy_per_channel[1]}")
    print(f"Red channel entropy: {entropy_per_channel[2]}")

    #Ex 8
    kernel = np.ones((5,5),np.float32)/25
    dst = cv.filter2D(img1,-1,kernel)

    plt.imshow(dst)
    plt.show()

    entropy_per_channel = entropy(dst)

    print(f"Blue channel entropy: {entropy_per_channel[0]}")
    print(f"Green channel entropy: {entropy_per_channel[1]}")
    print(f"Red channel entropy: {entropy_per_channel[2]}")



if __name__ == "__main__":
    main()