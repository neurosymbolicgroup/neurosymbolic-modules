package org.genesys.interpreter;

/**
 * Created by yufeng on 5/31/17.
 */
public class PrimitiveBinop implements Binop {
    private final String op;

    public PrimitiveBinop(String op) {
        this.op = op;
    }

    public Object apply(Object first, Object second) {
        if (this.op.equals("+")) {
            return (int) first + (int) second;
        } else if (this.op.equals("^")) {
            int res = (int)first;
            for (int i = 1; i <= (int)second; i++){
                res *= (int)first;
            }
            return res;
        } else if (this.op.equals("*")) {
            return (int) first * (int) second;
        } else if (this.op.equals(">")) {
            return (int) first > (int) second;
        } else if (this.op.equals(">=")) {
            return (int) first >= (int) second;
        } else if (this.op.equals("<")) {
            return (int) first < (int) second;
        } else if (this.op.equals("<=")) {
            return (int) first <= (int) second;
        } else if (this.op.equals("||")) {
            return (boolean) first || (boolean) second;
        } else if (this.op.equals("&&")) {
            return (boolean) first && (boolean) second;
        } else if (this.op.equals("-")) {
            return (int) first - (int) second;
        } else if (this.op.equals("/")) {
            return (int) first / (int) second;
        } else if (this.op.equals("~")) {
            return !(boolean) first;
        } else if (this.op.equals("%")) {
            return (int) first % (int) second;
        } else if (this.op.equals("==")) {
            return (int) first == (int) second;
        } else if (this.op.equals("!=")) {
            return (int) first != (int) second;
        } else if (this.op.equals("%=")){
            return (((int) first) % (int)second) == 0;
        } else if (this.op.equals("%!=")){
            return (((int) first) % (int)second) != 0;
        } else if (this.op.equals("%!=2")) {//ODD
            return (((int) first) % 2) != (int) second;
        } else if (this.op.equals("%=2")) {//EVEN
            return (((int) first) % 2) == (int) second;
        } else if (this.op.equals("**")) {
            return ((int) first * (int)first);
        } else {
            assert false : this.op;
            throw new RuntimeException();
        }
    }

    @Override
    public String toString() {
        return "l(a,b).(" + this.op + " a b)";
    }
}
