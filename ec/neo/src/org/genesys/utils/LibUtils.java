package org.genesys.utils;

import com.microsoft.z3.BoolExpr;
import org.genesys.type.AbstractList;
import org.genesys.type.Cons;
import org.genesys.type.EmptyList;

import java.lang.reflect.Array;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by yufeng on 5/29/17.
 */
public class LibUtils {

    /**
     * Convert list to array.
     * FIXME:How to implement the generic version?
     *
     * @param list
     * @param <T>
     * @return array
     */
    public static BoolExpr[] listToArray(List<BoolExpr> list) {
        BoolExpr[] array = new BoolExpr[list.size()];
        array = list.toArray(array);
        return array;
    }

    public static double computeTime(long start, long end) {
        double diff = end - start;
        return (diff / 1e6);
    }

    public static long tick() {
        return System.nanoTime();
    }

    public static AbstractList getAbsList(List arg) {
        LinkedList myList = new LinkedList(arg);
        AbstractList abstractList = construct(myList);
//        System.out.println("convert: " + abstractList);
        return abstractList;
    }

    public static Object fixGsonBug(Object data) {
        if (data instanceof List) {
            List dataList = (List) data;
            List tgtList = new ArrayList();
            for (Object l : dataList) {
                List innerList = new ArrayList();
                if (l instanceof ArrayList) {
                    for (Object elem : (List) l) {
                        if (elem instanceof Double) {
                            innerList.add(((Double) elem).intValue());
                        } else {
                            innerList.add(elem);
                        }
                    }
                    tgtList.add(innerList);
                } else {
                    if (l instanceof Double) {
                        tgtList.add(((Double) l).intValue());
                    } else {
                        tgtList.add(l);
                    }
                }
            }
            return tgtList;
        } else if (data instanceof Double) {
            return ((Double) data).intValue();
        } else {
            return data;
        }
    }

    /* recursively construct cons */
    private static AbstractList construct(LinkedList arg) {
        if (arg.isEmpty())
            return new EmptyList();
        else {
            Object fst = arg.pollFirst();
            //FIXME: The stupid bug in Gson.
            if (fst instanceof Double) fst = ((Double) fst).intValue();
            return new Cons(fst, construct(arg));
        }
    }

    public static List<String> extractNums(String str) {
        List list = new ArrayList();
        Pattern p = Pattern.compile("-?\\d+");
        Matcher m = p.matcher(str);
        while (m.find()) {
            list.add(m.group());
        }
        return list;
    }

    @SuppressWarnings("unchecked")
    public static <T extends List<?>> T cast(Object obj) {
        return (T) obj;
    }

    public static <X> X deepClone(final X input) {
        if (input == null) {
            return input;
        } else if (input instanceof Map<?, ?>) {
            return (X) deepCloneMap((Map<?, ?>) input);
        } else if (input instanceof Collection<?>) {
            return (X) deepCloneCollection((Collection<?>) input);
        } else if (input instanceof Object[]) {
            return (X) deepCloneObjectArray((Object[]) input);
        } else if (input.getClass().isArray()) {
            return (X) clonePrimitiveArray((Object) input);
        }

        return input;
    }

    private static Object clonePrimitiveArray(final Object input) {
        final int length = Array.getLength(input);
        final Object copy = Array.newInstance(input.getClass().getComponentType(), length);
        // deep clone not necessary, primitives are immutable
        System.arraycopy(input, 0, copy, 0, length);
        return copy;
    }

    private static <E> E[] deepCloneObjectArray(final E[] input) {
        final E[] clone = (E[]) Array.newInstance(input.getClass().getComponentType(), input.length);
        for (int i = 0; i < input.length; i++) {
            clone[i] = deepClone(input[i]);
        }

        return clone;
    }

    private static <E> Collection<E> deepCloneCollection(final Collection<E> input) {
        Collection<E> clone;
        // this is of course far from comprehensive. extend this as needed
        if (input instanceof LinkedList<?>) {
            clone = new LinkedList<E>();
        } else if (input instanceof SortedSet<?>) {
            clone = new TreeSet<E>();
        } else if (input instanceof Set) {
            clone = new HashSet<E>();
        } else {
            clone = new ArrayList<E>();
        }

        for (E item : input) {
            clone.add(deepClone(item));
        }

        return clone;
    }

    private static <K, V> Map<K, V> deepCloneMap(final Map<K, V> map) {
        Map<K, V> clone;
        // this is of course far from comprehensive. extend this as needed
        if (map instanceof LinkedHashMap<?, ?>) {
            clone = new LinkedHashMap<K, V>();
        } else if (map instanceof TreeMap<?, ?>) {
            clone = new TreeMap<K, V>();
        } else {
            clone = new HashMap<K, V>();
        }

        for (Map.Entry<K, V> entry : map.entrySet()) {
            clone.put(deepClone(entry.getKey()), deepClone(entry.getValue()));
        }

        return clone;
    }

    public static BitSet fromBitString(final String s) {
        return BitSet.valueOf(new long[]{Long.parseLong(s, 2)});
    }

    public static String toBitString(BitSet bs) {
        if(bs.isEmpty()) return "00000";
        return Long.toString(bs.toLongArray()[0], 2);
    }

    public static void main(String[] args) {
        BitSet mask = LibUtils.fromBitString("0000");
        BitSet b1 = LibUtils.fromBitString("1010");

        BitSet c1 = LibUtils.fromBitString("1100");
        BitSet c2 = LibUtils.fromBitString("0110");
        BitSet c3 = LibUtils.fromBitString("0011");
        System.out.println(mask);
        System.out.println(b1);
        System.out.println(c1);
        System.out.println(c2);
        System.out.println(c3);
        c1.and(b1);
        System.out.println("new c1:" + toBitString(c1));
        c2.and(b1);
        System.out.println("new c2:" + toBitString(c2));
        mask.or(c1);
        System.out.println("mask1:" + toBitString(mask));
        mask.or(c2);
        System.out.println("mask2:" + toBitString(mask) + " " + mask);
    }

}
