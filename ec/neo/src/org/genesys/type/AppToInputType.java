package org.genesys.type;

/**
 * Created by yufeng on 6/4/17.
 * I don't like this one.
 */
public class AppToInputType implements AbstractType {

    public final FunctionType functionType;

    public AppToInputType(FunctionType functionType) {
        this.functionType = functionType;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof AppToInputType)) {
            return false;
        }
        AppToInputType type = (AppToInputType) obj;
        return this.functionType.equals(type.functionType);
    }

    @Override
    public int hashCode() {
        return 5 * this.functionType.hashCode() + 4;
    }
}
