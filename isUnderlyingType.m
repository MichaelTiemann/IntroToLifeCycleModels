function ans = isUnderlyingType(x,type)
    ans = strcmp(class(x),type);
end
