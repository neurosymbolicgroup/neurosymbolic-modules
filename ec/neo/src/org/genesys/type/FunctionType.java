package org.genesys.type;

/**
 * Created by yufeng on 5/31/17.
 */
public class FunctionType implements AbstractType {
    public final AbstractType inputType;
    public final AbstractType outputType;

    public FunctionType(AbstractType inputType, AbstractType outputType) {
        this.inputType = inputType;
        this.outputType = outputType;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof FunctionType)) {
            return false;
        }
        FunctionType type = (FunctionType) obj;
        return this.inputType.equals(type.inputType) && this.outputType.equals(type.outputType);
    }

    @Override
    public int hashCode() {
        return 5 * (5 * this.inputType.hashCode() + this.outputType.hashCode()) + 3;
    }

    @Override
    public String toString() {
        return "(" + this.inputType.toString() + " -> " + this.outputType.toString() + ")";
    }
}
