import React from 'react';
import { createStackNavigator } from 'react-navigation-stack';
import { createBottomTabNavigator } from 'react-navigation-tabs';

import TabBarIcon from '../components/TabBarIcon';
import PricesScreen from '../screens/PricesScreen';
import CameraScreen from "../screens/CameraScreen";

const PriceListStack = createStackNavigator({
    Prices: PricesScreen,
    }
);

PricesScreen.navigationOptions = {
    header: null,
};

PriceListStack.navigationOptions = {
    tabBarLabel: "Prices",
    tabBarIcon: ({ focused }) => (
        <TabBarIcon focused={focused} name={Platform.OS === 'ios' ? 'ios-list' : 'md-list'}/>
    )
};

PriceListStack.path = "";

const CameraStack = createStackNavigator({
        Camera: CameraScreen,
    }
);

CameraScreen.navigationOptions = {
    header: null,
};

CameraStack.navigationOptions = {
    tabBarLabel: "Camera",
    tabBarIcon: ({ focused }) => (
        <TabBarIcon focused={focused} name={Platform.OS === 'ios' ? 'ios-camera' : 'md-camera'}/>
    )
};

CameraStack.path = "";

const tabNavigator = createBottomTabNavigator({
  PriceListStack,
  CameraStack,
});

tabNavigator.path = '';

export default tabNavigator;
