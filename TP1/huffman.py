
#from multiprocessing import Queue
from queue import PriorityQueue
from collections import Counter
import matplotlib.pyplot as plt
import cv2 as cv

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

def compress_img(huffman_dict):

    final_bit_str: str = f"{len(huffman_dict.keys()):08b}"
    #final_bit_str: str = f"{255:08b}"
    code_bits_dimension = int(len(bin(max(list(map(lambda x: len(x), huffman_dict.values()))))) - 2)
    final_bit_str += f"{code_bits_dimension:08b}"
    '''
    print(sorted_list)

    for i in sorted_list:
        final_bit_str += f"|{i:08b}{len(huffman_dict.get(i)):0{code_bits_dimension}b}"
    '''
    print(final_bit_str)

def main():
    img = cv.imread("./aula1.bmp", 0) 
    """ plt.imshow(img, cmap="gray")
    plt.show() """

    hist = cv.calcHist([img],[0],None,[256],[0,256])
    """ plt.plot(hist), plt.xlim([0, 256])
    plt.show() """
    huffman_dict = huffman(img.flatten())
    print(huffman_dict)

    compress_img(huffman_dict)


if __name__ == "__main__":
    main()